"""Frozen OpenQARP full-statevector checks, not a BBO experiment."""
from __future__ import annotations
import hashlib
import importlib.metadata
import itertools
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'src'))
from qarp_backend import OpenQARPXYQAOA, all_bitstrings_lsb, one_hot_penalty


def save(name, obj):
    (ROOT/'json'/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_basis(sizes):
    labels = list(itertools.product(*(range(k) for k in sizes)))
    bits = np.zeros((len(labels), sum(sizes)), dtype=np.int8)
    offsets = np.cumsum([0]+list(sizes[:-1]))
    for i, label in enumerate(labels):
        for g, choice in enumerate(label):bits[i, offsets[g]+choice] = 1
    return labels, bits, {label:i for i,label in enumerate(labels)}


def independent_edges(sizes):
    edges=[];offset=0
    for group, size in enumerate(sizes):
        local=[(k,k+1) for k in range(size-1)]
        if size>2:local.append((size-1,0))
        edges.extend((group,a,b,offset+a,offset+b) for a,b in local)
        offset+=size
    return edges


def reference_evolution(q, sizes, gamma, beta):
    labels, bits, lookup=independent_basis(sizes)
    energy=np.einsum('bi,ij,bj->b',bits,q,bits)
    state=np.exp(-1j*gamma*energy)/np.sqrt(len(bits))
    for group, a, b, _, _ in independent_edges(sizes):
        for index,label in enumerate(labels):
            if label[group]!=a:continue
            other=list(label);other[group]=b;j=lookup[tuple(other)]
            left,right=state[index],state[j]
            state[index]=np.cos(beta)*left-1j*np.sin(beta)*right
            state[j]=np.cos(beta)*right-1j*np.sin(beta)*left
    return state,bits,energy


def invalid_mass(state,sizes):
    # Scan the full space in bounded chunks; no full 2**N-by-N bit matrix.
    total=0.;largest=0.
    for start in range(0,len(state),65536):
        stop=min(start+65536,len(state));indices=np.arange(start,stop,dtype=np.uint64)
        valid=np.ones(stop-start,dtype=bool);offset=0
        for size in sizes:
            count=np.zeros(stop-start,dtype=np.uint8)
            for bit in range(offset,offset+size):count+=((indices>>bit)&1).astype(np.uint8)
            valid &= count==1;offset+=size
        probabilities=np.abs(state[start:stop][~valid])**2
        total+=float(probabilities.sum())
        if len(probabilities):largest=max(largest,float(probabilities.max()))
    return total,largest


def run_case(sizes, seed, protocol):
    n=sum(sizes);rng=np.random.default_rng(seed)
    q=np.diag(rng.uniform(-1,1,n))
    for i in range(n-1):q[i,i+1]=rng.uniform(-1,1)
    backend=OpenQARPXYQAOA(q,sizes)
    edges=[(a,b) for _,_,_,a,b in independent_edges(sizes)]
    if backend.mixer_edges!=edges:raise AssertionError('Ring edge/order mismatch')
    _,bits,_=independent_basis(sizes)
    expected_indices=bits.astype(np.uint64)@(1<<np.arange(n,dtype=np.uint64))
    np.testing.assert_array_equal(backend.feasible_indices,expected_indices)
    initial=backend.initial_state[expected_indices]
    initial_norm=float(np.vdot(backend.initial_state,backend.initial_state).real)
    fidelity=float(abs(np.vdot(np.ones(len(bits))/np.sqrt(len(bits)),initial))**2)
    assert abs(1-initial_norm)<1e-10 and abs(1-fidelity)<1e-10
    # Exact Ising energy offset for an upper-triangular binary QUBO.
    z=1-2*bits.astype(float)
    ising=np.zeros(len(bits))
    for i,c in backend.linear.items():ising+=c*z[:,i]
    for (i,j),c in backend.quadratic.items():ising+=c*z[:,i]*z[:,j]
    offset=float(np.trace(q)/2+np.triu(q,1).sum()/4)
    direct=np.einsum('bi,ij,bj->b',bits,q,bits)
    energy_error=float(np.max(np.abs(ising+offset-direct)))
    assert energy_error<1e-10
    penalty=None
    if n<=9:
        all_bits=all_bitstrings_lsb(n);p=one_hot_penalty(all_bits,sizes)
        mask=np.ones(len(all_bits),dtype=bool);pos=0
        for k in sizes:mask &= all_bits[:,pos:pos+k].sum(axis=1)==1;pos+=k
        assert np.all(p[mask]==0) and np.all(p[~mask]>0)
        all_z=1-2*all_bits;all_ising=np.full(len(all_bits),offset)
        for i,c in backend.linear.items():all_ising+=c*all_z[:,i]
        for (i,j),c in backend.quadratic.items():all_ising+=c*all_z[:,i]*all_z[:,j]
        all_direct=np.einsum('bi,ij,bj->b',all_bits,q,all_bits)
        full_energy_error=float(np.max(np.abs(all_ising-all_direct)))
        assert full_energy_error<1e-10
        penalty={'all_basis_states_checked':len(all_bits),'max_feasible_penalty':float(p[mask].max()),'min_invalid_penalty':float(p[~mask].min()),'all_basis_qubo_ising_max_error':full_energy_error}
    result={'n_qubits':n,'group_sizes':sizes,'seed':seed,'qubo':q.tolist(),'mixer_edges':[list(e) for e in edges],
      'initial_norm':initial_norm,'initial_fidelity_to_product_w':fidelity,'feasible_basis_count':len(bits),
      'qubo_ising_energy_max_error':energy_error,'omitted_global_energy_offset':offset,'penalty_check':penalty,'angle_results':[]}
    for gamma,beta in protocol['angle_pairs']:
        t=time.perf_counter();state=backend.statevector([gamma],[beta]);elapsed=time.perf_counter()-t
        norm=float(np.vdot(state,state).real)
        feasible_state=np.array(state[expected_indices],copy=True)
        p_feasible=float(np.vdot(feasible_state,feasible_state).real)
        leak,max_invalid=invalid_mass(state,sizes)
        expected,_,_=reference_evolution(q,sizes,gamma,beta)
        # Circuit omits a scalar Ising offset; remove only that global phase.
        expected *= np.exp(1j*gamma*offset)
        error=float(np.max(np.abs(feasible_state-expected)))
        drift=float(np.max(np.abs(np.abs(feasible_state)**2-1/len(bits))))
        if gamma!=0 and beta!=0:assert drift>1e-6, 'Diagnostic evolution unexpectedly uniform'
        check={'gamma':gamma,'beta':beta,'raw_norm':norm,'p_feasible_raw':p_feasible,'feasibility_error':abs(1-p_feasible),
          'invalid_probability_mass':leak,'max_invalid_basis_probability':max_invalid,'reference_amplitude_max_error':error,
          'max_probability_deviation_from_uniform':drift,'statevector_seconds':elapsed,
          'status':'passed' if abs(1-norm)<1e-10 and abs(1-p_feasible)<1e-10 and leak<1e-10 and error<1e-10 else 'failed'}
        result['angle_results'].append(check)
        print(json.dumps({'n':n,'seed':seed,**check}),flush=True)
        save('circuit_validation_progress.json',result)
        del state
        if check['status']!='passed':raise AssertionError('Statevector validation failed')
    result['status']='passed'
    return result


def main():
    path=ROOT/'json/circuit_validation_protocol.json';protocol=json.loads(path.read_text())
    if protocol['status']!='ready_for_circuit_prevalidation_only' or protocol['fm_training_or_bbo_allowed']:
        raise RuntimeError('Review the circuit-only protocol first')
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'py'),'-p','test_xy_ring.py','-v'],capture_output=True,text=True)
    (ROOT/'md/circuit_validation_tests.log').write_text(tests.stdout+tests.stderr)
    results={'status':'running','scope':'ideal_gate_circuit_prevalidation_only','protocol_sha256':digest(path),
             'backend':'OpenQARP/qarpx','onehot_penalty_lambda':0,'p':1,'unit_test_exit_code':tests.returncode,'cases':[]}
    try:
        if tests.returncode:raise AssertionError('Circuit regression tests failed')
        for sizes in protocol['cases']:
            for seed in protocol['qubo_seeds']:
                results['cases'].append(run_case(sizes,seed,protocol));save('circuit_validation.json',results)
        results['status']='passed'
    except Exception:
        results['status']='failed';results['failure']=traceback.format_exc();raise
    finally:
        save('circuit_validation.json',results)
        files=[REPO/'src/qarp_backend.py',REPO/'src/qaoa_solver.py',ROOT/'py/validate_xy_ring.py',ROOT/'py/test_xy_ring.py']
        save('circuit_validation_environment.json',{'time_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,
          'executable':sys.executable,'platform':platform.platform(),'versions':{p:importlib.metadata.version(p) for p in ['openqarp','numpy','scipy']},
          'command':sys.executable+' intern_1009/perovskites/py/validate_xy_ring.py','code_sha256':{str(p.relative_to(REPO)):digest(p) for p in files},
          'initial_state_vector_bytes_n23':(1<<23)*16,'no_full_2_to_n_by_n_bit_matrix':True})
    print('Circuit prevalidation passed',flush=True)


if __name__=='__main__':main()
