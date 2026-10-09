"""Small restrictions of existing FM models: angle-grid and Cost-scale audit."""
import sys,json,time,platform,itertools
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parents[1];PILOT=OUT.parent/'fixed_fm_pilot'
sys.path.insert(0,str(PILOT/'py'))
from run_pilot import (sha,save,matrix,build_penalty_bqm,OpenQARPStandardQAOA,OpenQARPXYQAOA,qubo_to_ising_coefficients,REPO)
from qarp_backend import all_bitstrings_lsb,one_hot_penalty

def restrict(q,offset,free,fixed):
    q=np.asarray(q);fixed=np.asarray(fixed);sub=q[np.ix_(free,free)].copy()
    for k,i in enumerate(free):
        sub[k,k]+=sum(q[min(i,j),max(i,j)] for j in np.flatnonzero(fixed) if j!=i)
    return sub,float(offset+fixed@q@fixed)

def batch_best(prob,values,shots=10):
    order=np.argsort(values,kind='stable');p=np.array(prob)[order];f=np.array(values)[order]
    before=np.concatenate([[0.],np.cumsum(p)[:-1]]);after=np.cumsum(p)
    weight=np.maximum(0.,(1-before)**shots-(1-after)**shots)
    success=float(weight.sum());return (float(weight@(f-f.min())/success) if success>0 else None),success

def scale(q):
    h,j=qubo_to_ising_coefficients(q);return max([abs(v) for v in h.values()]+[abs(v) for v in j.values()]+[0.])

def main():
    protocol=json.loads((OUT/'json/protocol.json').read_text());assert protocol['status']=='ready'
    validation=json.loads((OUT/'json/prevalidation.json').read_text());assert validation['status']=='passed' and validation['runner_sha256']==sha(Path(__file__))
    assert json.loads((PILOT/'json/artifact_verification.json').read_text())['status']=='passed'
    hashes={str(p.relative_to(REPO)):sha(p) for p in [Path(__file__),REPO/'src/qarp_backend.py',REPO/'src/fmqa_solver.py']}
    save(OUT/'json/environment.json',{'python':sys.version,'platform':platform.platform(),'protocol_sha256':sha(OUT/'json/protocol.json'),'code_sha256':hashes})
    for seed in protocol['seeds']:
        target=OUT/f'json/seed_{seed}.json'
        if target.exists():raise RuntimeError('existing data preserved; use new folder for rerun')
        source=PILOT/f'json/seed_{seed}.json';base=json.loads(source.read_text());assert base['status']=='completed'
        assert sha(PILOT/base['checkpoint'])==base['checkpoint_sha256']
        record={'seed':seed,'status':'running','source_sha256':sha(source),'model_sha256':base['fixed_model_sha256'],
          'initial_sha256':base['initial_sha256'],'checkpoint_sha256':base['checkpoint_sha256'],
          'protocol_sha256':sha(OUT/'json/protocol.json'),'variants':[]};save(target,record)
        for variant in protocol['variants']:
            free=variant['free_indices'];fixed=np.zeros(23);fixed[variant['fixed_one_indices']]=1
            q,offset=restrict(base['qubo'],base['offset'],free,fixed);bits=all_bitstrings_lsb(len(free))
            full=np.tile(fixed,(len(bits),1));full[:,free]=bits
            fm_values=np.einsum('bi,ij,bj->b',bits,q,bits)+offset
            full_values=np.einsum('bi,ij,bj->b',full,np.array(base['qubo']),full)+base['offset']
            np.testing.assert_allclose(fm_values,full_values,atol=1e-10)
            penalty=one_hot_penalty(bits,variant['groups']);valid=penalty==0;fvalid=fm_values[valid];s=scale(q);div=s if s>0 else 1.
            vrecord={'variant_id':variant['id'],'n':len(free),'free_indices':free,'fixed_one_indices':variant['fixed_one_indices'],
              'group_sizes':variant['groups'],'restricted_qubo':q.tolist(),'offset':offset,'base_Ising_S':s,
              'full_state_count':len(bits),'feasible_state_count':int(valid.sum()),'restriction_max_error':float(np.max(np.abs(fm_values-full_values))), 'settings':[]}
            cases=[('XY-FMQAOA',0,'base_S')]+[('Penalty-FMQAOA',a,m) for a in protocol['alphas'] for m in protocol['standard_modes']]
            for method,alpha,mode in cases:
                qdict={(i,j):q[i,j]/div for i in range(len(free)) for j in range(i,len(free))}
                bqm=build_penalty_bqm(qdict,variant['groups'],s*alpha/div,offset/div);qd,total_offset=bqm.to_qubo();total_q=matrix(qd,len(free))
                cost_values=(fm_values+s*alpha*penalty)/div
                cost_divisor=scale(total_q) if mode=='full_Cost_S' else 1.;cost_divisor=cost_divisor if cost_divisor>0 else 1.
                if method=='XY-FMQAOA':backend=OpenQARPXYQAOA(q/div,variant['groups'])
                else:backend=OpenQARPStandardQAOA(q/div/cost_divisor,variant['groups'],s*alpha/div/cost_divisor)
                optimum=float(cost_values.min());ground_mask=np.abs(cost_values-optimum)<1e-9
                for search in protocol['searches']:
                    gammas=search.get('gammas',np.linspace(0,2*np.pi,search.get('axis_points',3)).tolist())
                    betas=search.get('betas',np.linspace(0,np.pi,search.get('axis_points',3)).tolist())
                    points=[];best=None;best_prob=None;start=time.perf_counter()
                    for gamma,beta in itertools.product(gammas,betas):
                        state=backend.statevector([gamma],[beta]);prob=np.abs(state)**2;norm=float(prob.sum());assert abs(norm-1)<1e-10;prob/=norm
                        feasible=float(prob[valid].sum())
                        if method=='XY-FMQAOA':assert abs(feasible-1)<1e-10 and alpha==0
                        expectation=float(prob@cost_values)
                        points.append({'gamma':gamma,'beta':beta,'raw_norm':norm,'expected_common_baseS_Cost':expectation,'feasible_probability':feasible})
                        if best is None or expectation<best['expected_common_baseS_Cost']:
                            best=points[-1];best_prob=prob.copy()
                    assert len(points)==search['budget']
                    pf=float(best_prob[valid].sum());popt=float(best_prob[valid & (np.abs(fm_values-fvalid.min())<1e-9)].sum())
                    gap10,success10=batch_best(best_prob[valid],fvalid,10)
                    item={'method':method,'alpha':alpha,'mode':mode,'search':search['id'],'budget':len(points),
                      'lambda_internal':s*alpha,'Cost_divisor':cost_divisor,'gamma_effective_original':best['gamma']/cost_divisor,
                      'selected_angles':[best['gamma'],best['beta']],'expected_common_baseS_Cost':best['expected_common_baseS_Cost'],
                      'expectation_gap_to_penalized_ground':best['expected_common_baseS_Cost']-optimum,
                      'penalized_ground_value':optimum,'all_penalized_ground_states_feasible':bool(np.all(valid[ground_mask])),
                      'feasible_probability':pf,'fm_optimum_probability':popt,'success10':success10,
                      'fm_optimum_success10':float(-np.expm1(10*np.log1p(-min(popt,1-1e-15)))) if popt<1 else 1.,
                      'conditional_best10_FM_gap_eV':gap10,'max_feasibility_in_evaluated_grid':max(p['feasible_probability'] for p in points),
                      'selected_basis_probabilities':best_prob.tolist(),'angle_search':points,'generation_seconds':time.perf_counter()-start}
                    assert abs(success10-(1-(1-pf)**10))<1e-10
                    vrecord['settings'].append(item)
                print(f'finished seed={seed} {variant["id"]} {method} alpha={alpha} mode={mode}',flush=True)
            record['variants'].append(vrecord);save(target,record)
        record['status']='completed';save(target,record)
    print('completed all small-instance angle validations',flush=True)

if __name__=='__main__':main()
