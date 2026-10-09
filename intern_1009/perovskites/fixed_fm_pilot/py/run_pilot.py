"""Reproducible fixed-FM diagnostic; not a closed-loop BBO runner."""
from __future__ import annotations
import hashlib
import itertools
import json
import platform
import sys
import time
import traceback
from pathlib import Path
import importlib.metadata
import numpy as np
import torch
from scipy.stats import spearmanr

OUT=Path(__file__).resolve().parents[1]
PROBLEM=OUT.parent
REPO=PROBLEM.parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src'),str(PROBLEM/'py')]
from perovskites_evaluator import PerovskitesEvaluator
from fm import train_factorization_machine
from fm_to_qubo import fm_to_qubo
from fmqa_solver import build_penalty_bqm
from qarp_backend import OpenQARPXYQAOA, OpenQARPStandardQAOA, qubo_to_ising_coefficients
from intern_0924.src.penalty import AdaptivePenaltyTracker
from intern_0924.src.solvers import _build_qubo_energies, _state_energy_expectation, _sample_statevector_streaming
import neal


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def model_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def matrix(qdict,n):
    q=np.zeros((n,n))
    for (i,j),v in qdict.items():q[min(i,j),max(i,j)]+=v
    return q

def finite_float(value):return float(value) if np.isfinite(value) else None


def sa_generate(qdict,offset,scale,sizes,seed,adaptive,protocol):
    tracker=AdaptivePenaltyTracker(**{'initial_alpha':protocol['adaptive']['initial_alpha'],'min_alpha':0.01,'max_alpha':100})
    pools=[];batches=[];start=time.perf_counter()
    for batch in range(protocol['sa']['batches']):
        alpha=tracker.get_current_alpha() if adaptive else protocol['large_penalty_alpha']
        lam=scale*alpha
        bqm=build_penalty_bqm(qdict,list(sizes),lam,offset)
        samples=neal.SimulatedAnnealingSampler().sample(bqm,num_reads=250,num_sweeps=1000,seed=seed*100+batch,beta_schedule_type='geometric')
        arr=samples.record.sample[:,[list(samples.variables).index(i) for i in range(sum(sizes))]]
        bits=np.repeat(arr,samples.record.num_occurrences,axis=0).astype(np.int8)
        pos=0;feasible=np.ones(len(bits),dtype=bool)
        for k in sizes:feasible &= bits[:,pos:pos+k].sum(axis=1)==1;pos+=k
        rate=float(feasible.mean());pools.append(bits)
        batches.append({'batch':batch,'alpha':alpha,'lambda_internal':lam,'base_scale':scale,'raw_feasible_rate':rate,'sample_count':len(bits),'next_alpha':tracker.update(rate) if adaptive else alpha,'neal_info':{k:v for k,v in samples.info.items() if k!='timing'}})
    return np.concatenate(pools),{'batches':batches,'raw_feasible_rate_kind':'empirical_SA','generation_seconds':time.perf_counter()-start}


def qaoa_generate(q,offset,scale,sizes,seed,xy,protocol,valid_indices):
    divisor=scale if scale>0 else 1.0
    normalized=q/divisor;bias=offset/divisor
    alpha=0.0 if xy else protocol['penalty_qaoa_alpha']
    normalized_lambda=scale*alpha/divisor
    backend=OpenQARPXYQAOA(normalized,sizes) if xy else OpenQARPStandardQAOA(normalized,sizes,normalized_lambda)
    qdict={(i,j):normalized[i,j] for i in range(len(q)) for j in range(i,len(q))}
    bqm=build_penalty_bqm(qdict,list(sizes),normalized_lambda,bias)
    total_dict,total_offset=bqm.to_qubo()
    total_q=matrix(total_dict,len(q))
    t=time.perf_counter();energies=_build_qubo_energies(total_q,total_offset,np.dtype(np.float64))
    best_state=None;best_value=float('inf');search=[]
    for gamma,beta in itertools.product(protocol['qaoa']['gammas'],protocol['qaoa']['betas']):
        state=backend.statevector([gamma],[beta])
        value,norm=_state_energy_expectation(state,energies,65536)
        if abs(norm-1)>1e-10:raise RuntimeError('raw norm drift')
        search.append({'gamma':gamma,'beta':beta,'expected_normalized_surrogate_plus_penalty':value,'raw_norm':norm})
        print(f'seed={seed} method={"XY" if xy else "X"} angle={gamma},{beta} surrogate={value:.6g}',flush=True)
        if value<best_value:
            best_value=value;best_state=state;best_angles=[gamma,beta]
        else:del state
    del energies
    indices=_sample_statevector_streaming(best_state,protocol['qaoa']['shots'],seed,65536)
    bits=((indices[:,None]>>np.arange(len(q),dtype=np.uint64)[None,:])&1).astype(np.int8)
    valid_prob=np.abs(best_state[valid_indices])**2
    meta={'angles':best_angles,'angle_search':search,'lambda_internal':scale*alpha,'alpha':alpha,'base_scale':scale,
          'normalization_divisor':divisor,'raw_feasible_rate_kind':'exact_ideal_statevector','exact_feasible_probability':float(valid_prob.sum()),
          'feasible_basis_probabilities':valid_prob.tolist(),'generation_seconds':time.perf_counter()-t,
          'full_statevector':True,'float64_energy_array_bytes':8*(1<<len(q)),'full_bit_matrix_allocated':False}
    del best_state,backend
    return bits,meta


def score_candidates(bits,meta,q,offset,categories,patterns,train_ids,truth,protocol):
    lookup={tuple(row):i for i,row in enumerate(patterns)}
    ids=[lookup.get(tuple(row)) for row in bits]
    train_set=set(train_ids);counts={};invalid=0;seen_initial=0;pool_duplicates=0
    for idx in ids:
        if idx is None:invalid+=1;continue
        if idx in train_set:seen_initial+=1;continue
        if idx in counts:pool_duplicates+=1
        counts[idx]=counts.get(idx,0)+1
    predictions=np.einsum('bi,ij,bj->b',patterns,q,patterns)+offset
    ranked=sorted(counts,key=lambda i:(predictions[i],tuple(patterns[i])))
    accepted=ranked[:protocol['accepted_candidates']]
    initial_best=float(np.min(truth[train_ids]));new_best=float(np.min(truth[accepted])) if accepted else None
    final_best=min(initial_best,new_best) if new_best is not None else initial_best
    fmin,fmax=float(truth.min()),float(truth.max());den=fmax-fmin
    n=len(bits)
    empirical_feas=(n-invalid)/n;unseen_shots=sum(counts.values())
    if 'feasible_basis_probabilities' in meta:
        prob=np.array(meta['feasible_basis_probabilities']);unseen=np.ones(len(prob),bool);unseen[train_ids]=False
        meta['exact_unseen_feasible_probability']=float(prob[unseen].sum())
        mass=meta['exact_unseen_feasible_probability']
        meta['exact_conditional_unseen_expected_true_gap']=float(prob[unseen]@truth[unseen]/mass) if mass>0 else None
        meta['exact_unseen_true_top5pct_probability']=float(prob[unseen & (truth<=np.sort(truth)[9])].sum())
        if meta['alpha']==0 and abs(1-meta['exact_feasible_probability'])>1e-10:raise RuntimeError('XY feasibility failed')
    output={**meta,'raw_sample_count':n,'empirical_raw_feasible_rate':empirical_feas,'raw_unseen_feasible_fraction':unseen_shots/n,
      'empirical_conditional_unseen_mean_true_gap':float(sum(truth[i]*count for i,count in counts.items())/unseen_shots) if unseen_shots else None,
      'raw_invalid_count':invalid,'initial_duplicate_count':seen_initial,'within_unseen_pool_duplicate_count':pool_duplicates,
      'initial_duplicate_fraction':seen_initial/n,'within_unseen_pool_duplicate_fraction':pool_duplicates/n,
      'unique_unseen_candidates':len(counts),'accepted_count':len(accepted),'unused_new_bb_budget':protocol['accepted_candidates']-len(accepted),
      'bb_evaluations_initial':len(train_ids),'bb_evaluations_new':len(accepted),'allowed_total_bb_budget':25,
      'accepted_candidates':[{'candidate_id':i,'categories':categories[i],'onehot':patterns[i].tolist(),'fm_prediction':float(predictions[i]),'hse_gap':float(truth[i]),'raw_occurrences':counts[i],
                             'surrogate_gap_to_global_feasible_fm_min':float(predictions[i]-predictions.min())} for i in accepted],
      'initial_best_hse_gap':initial_best,'proposal_best_hse_gap':new_best,'best_initial_plus_proposals':final_best,
      'normalized_regret_initial_plus_proposals':(final_best-fmin)/den,
      'raw_basis_indices':[sum(int(b)<<q for q,b in enumerate(row)) for row in bits]}
    return output


def run_seed(seed,protocol,protocol_sha,code_hashes):
    bb=PerovskitesEvaluator();assert bb.data_sha256==protocol['data_sha256']
    categories=[list(c) for c in bb.candidates()];patterns=np.array([bb.encode(c) for c in categories],dtype=np.int8)
    train_ids=np.random.default_rng(seed).choice(len(categories),protocol['initial_samples'],replace=False).tolist()
    ytrain=np.array([bb.evaluate(categories[i]) for i in train_ids],dtype=np.float32)
    initial={'candidate_ids':train_ids,'records':[{'candidate_id':i,'categories':categories[i],'onehot':patterns[i].tolist(),'hse_gap':float(bb.evaluate(categories[i]))} for i in train_ids]}
    initial_sha=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest()
    torch.set_num_threads(1);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    t=time.perf_counter()
    model=train_factorization_machine(torch.tensor(patterns[train_ids],dtype=torch.float32),torch.tensor(ytrain),23,k=2,epochs=120,learning_rate=0.1)
    model.eval();training_seconds=time.perf_counter()-t;fixed_hash=model_hash(model)
    checkpoint=OUT/f'data/checkpoints/seed_{seed}.pt';torch.save({'state_dict':model.state_dict(),'initial_sha256':initial_sha,'protocol_sha256':protocol_sha,'seed':seed,'fm_config':protocol['fm']},checkpoint)
    qdict,offset=fm_to_qubo(model);q=matrix(qdict,23)
    h,j=qubo_to_ising_coefficients(q);scale=max([abs(v) for v in h.values()]+[abs(v) for v in j.values()]+[0.])
    with torch.no_grad():fm_values=model(torch.tensor(patterns,dtype=torch.float32)).cpu().numpy()
    qvalues=np.einsum('bi,ij,bj->b',patterns,q,patterns)+offset
    parity=float(np.max(np.abs(fm_values-qvalues)));assert parity<1e-4
    if not np.isfinite(q).all():raise RuntimeError('FM nonfinite coefficients')
    record={'seed':seed,'status':'running','protocol_sha256':protocol_sha,'code_sha256':code_hashes,'initial_sha256':initial_sha,'initial_dataset':initial,
       'fixed_model_sha256':fixed_hash,'checkpoint':str(checkpoint.relative_to(OUT)),'checkpoint_sha256':sha(checkpoint),
       'training_seconds':training_seconds,'qubo':q.tolist(),'offset':offset,'base_Ising_scale_S':scale,'fm_qubo_max_error':parity,'methods':{}}
    output=OUT/f'json/seed_{seed}.json';save(output,record)
    # True values below are audit-only, excluded from angle search/training/proposal ranking.
    truth=np.array([bb.evaluate(c) for c in categories]);test_ids=[i for i in range(len(categories)) if i not in set(train_ids)]
    record['fm_quality']={'train_count':len(train_ids),'test_count':len(test_ids),'train_rmse':float(np.sqrt(np.mean((qvalues[train_ids]-truth[train_ids])**2))),
        'test_rmse':float(np.sqrt(np.mean((qvalues[test_ids]-truth[test_ids])**2))),'test_spearman':finite_float(spearmanr(qvalues[test_ids],truth[test_ids]).statistic),
        'offline_reference_evaluations_not_solver_bb_budget':192,'all_candidate_predictions':qvalues.tolist()}
    record['uniform_unseen_reference']={'mean_true_gap':float(truth[test_ids].mean()),'true_top5pct_fraction':float(np.mean(truth[test_ids]<=np.sort(truth)[9])),
        'predicted_min_candidate_id':int(np.argmin(qvalues)),'predicted_min_true_gap':float(truth[np.argmin(qvalues)])}
    valid_indices=patterns.astype(np.uint64)@(1<<np.arange(23,dtype=np.uint64))
    for method in protocol['methods']:
        print(f'seed={seed} starting {method}',flush=True)
        t=time.perf_counter()
        if 'FMQAOA' in method:
            bits,meta=qaoa_generate(q,offset,scale,[16,3,4],seed,method=='XY-FMQAOA',protocol,valid_indices)
        else:bits,meta=sa_generate(qdict,offset,scale,[16,3,4],seed,method=='Adaptive-FMQA',protocol)
        assert len(bits)==1000 and model_hash(model)==fixed_hash
        result=score_candidates(bits,meta,q,offset,categories,patterns,train_ids,truth,protocol)
        result['initial_sha256']=initial_sha;result['fixed_model_sha256']=fixed_hash;result['method_total_seconds']=time.perf_counter()-t
        record['methods'][method]=result;save(output,record)
        print(f"seed={seed} finished {method}: feasible={result['empirical_raw_feasible_rate']:.4f} accepted={result['accepted_count']} best={result['best_initial_plus_proposals']:.4f}",flush=True)
    record['status']='completed';save(output,record)


def main():
    p=OUT/'json/protocol.json';protocol=json.loads(p.read_text());assert protocol['status']=='ready'
    base=json.loads((PROBLEM/'json/problem_manifest.json').read_text())
    assert base['evaluator']['validated'] and base['prevalidation']['target_group_sizes_passed'] and base['prevalidation']['legacy_n6_n9_passed']
    code=[Path(__file__),REPO/'src/fm.py',REPO/'src/fm_to_qubo.py',REPO/'src/fmqa_solver.py',REPO/'src/qarp_backend.py',REPO/'intern_0924/src/solvers.py',REPO/'intern_0924/src/penalty.py',PROBLEM/'py/perovskites_evaluator.py']
    hashes={str(f.relative_to(REPO)):sha(f) for f in code}
    save(OUT/'json/environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'versions':{p:importlib.metadata.version(p) for p in ['numpy','torch','scipy','openqarp','dwave-neal','dimod']},'protocol_sha256':sha(p),'code_sha256':hashes})
    failures=[]
    for seed in protocol['seeds']:
        target=OUT/f'json/seed_{seed}.json'
        if target.exists():
            old=json.loads(target.read_text())
            if old['status']=='completed' and old['protocol_sha256']==sha(p) and old['code_sha256']==hashes:
                print(f'reuse complete seed {seed}',flush=True);continue
            raise RuntimeError(f'existing unmatched or incomplete seed record: {seed}; preserve before rerun')
        try:run_seed(seed,protocol,sha(p),hashes)
        except Exception:
            details=traceback.format_exc();failures.append(seed)
            old=json.loads(target.read_text()) if target.exists() else {'seed':seed,'methods':{}}
            old.update({'status':'failed','failure':details,'protocol_sha256':sha(p),'code_sha256':hashes});save(target,old)
            print(details,flush=True)
    if failures:raise SystemExit(f'failed seeds retained: {failures}')
    print('All fixed-FM pilot seeds completed',flush=True)


if __name__=='__main__':main()
