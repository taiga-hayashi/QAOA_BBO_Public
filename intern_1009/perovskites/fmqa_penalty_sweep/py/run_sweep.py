"""Fixed-FM classical SA penalty sensitivity; no retraining or closed-loop BBO."""
import sys, json, hashlib, platform, time, traceback, importlib.metadata
from pathlib import Path
import numpy as np
import neal
from scipy.stats import wilcoxon

OUT=Path(__file__).resolve().parents[1]
PROBLEM=OUT.parent
REPO=PROBLEM.parents[1]
PILOT=PROBLEM/'fixed_fm_pilot'
sys.path[:0]=[str(PILOT/'py'),str(REPO/'src'),str(PROBLEM/'py'),str(REPO)]
from run_pilot import score_candidates, sha, save
from fmqa_solver import build_penalty_bqm
from qarp_backend import qubo_to_ising_coefficients
from perovskites_evaluator import PerovskitesEvaluator

def measure(bits, q, offset, patterns, train_ids, truth, categories):
    r=score_candidates(bits,{},q,offset,categories,patterns,train_ids,truth,{'accepted_candidates':5})
    lookup={tuple(p):i for i,p in enumerate(patterns)}
    ids=[lookup.get(tuple(b)) for b in bits]
    unseen=[i for i in ids if i is not None and i not in set(train_ids)]
    pred=np.einsum('bi,ij,bj->b',patterns,q,patterns)+offset
    fm_top=set(np.argsort(pred,kind='stable')[:10].tolist())
    true_top=set(np.argsort(truth,kind='stable')[:10].tolist())
    counts=np.array(list(__import__('collections').Counter(unseen).values()))
    probs=counts/counts.sum() if len(counts) else None
    r.update({
      'batch_any_feasible':int(any(i is not None for i in ids)),
      'batch_any_unseen_feasible':int(bool(unseen)),
      'batch_at_least_five_unseen':int(len(set(unseen))>=5),
      'batch_any_unseen_fm_top5pct':int(bool(set(unseen)&fm_top)),
      'batch_any_unseen_true_top5pct':int(bool(set(unseen)&true_top)),
      'raw_unseen_fm_top5pct_fraction':sum(i in fm_top for i in unseen)/len(bits),
      'raw_unseen_true_top5pct_fraction':sum(i in true_top for i in unseen)/len(bits),
      'unique_unseen_true_top5pct_count':len(set(unseen)&true_top),
      'conditional_unseen_mean_surrogate_gap':float(np.mean(pred[unseen]-pred.min())) if unseen else None,
      'conditional_unseen_effective_diversity':float(np.exp(-np.sum(probs*np.log(probs)))) if unseen else None,
      'best_unseen_surrogate_gap':float(np.min(pred[unseen])-pred.min()) if unseen else None,
    })
    return r

def main():
    p=OUT/'json/protocol.json';protocol=json.loads(p.read_text());assert protocol['status']=='ready'
    validation=json.loads((OUT/'json/prevalidation.json').read_text())
    assert validation['status']=='passed' and validation['runner_sha256']==sha(Path(__file__))
    audit=json.loads((PILOT/'json/artifact_verification.json').read_text())
    assert audit['status']=='passed'
    bb=PerovskitesEvaluator();assert bb.data_sha256==protocol['data_sha256']
    categories=[list(c) for c in bb.candidates()]
    patterns=np.array([bb.encode(c) for c in categories],dtype=np.int8)
    truth=np.array([bb.evaluate(c) for c in categories])
    sources=[Path(__file__),PILOT/'py/run_pilot.py',REPO/'src/fmqa_solver.py',PROBLEM/'py/perovskites_evaluator.py']
    hashes={str(f.relative_to(REPO)):sha(f) for f in sources}
    save(OUT/'json/environment.json',{'python':sys.version,'platform':platform.platform(),
      'versions':{k:importlib.metadata.version(k) for k in ['numpy','scipy','dwave-neal','dimod','torch']},
      'code_sha256':hashes,'protocol_sha256':sha(p)})
    failures=[]
    for seed in protocol['seeds']:
        target=OUT/f'json/seed_{seed}.json'
        if target.exists():raise RuntimeError('Preserve existing results; use a new directory for rerun')
        base=json.loads((PILOT/f'json/seed_{seed}.json').read_text())
        assert base['status']=='completed' and sha(PILOT/base['checkpoint'])==base['checkpoint_sha256']
        q=np.array(base['qubo']);offset=base['offset'];s=base['base_Ising_scale_S']
        h,j=qubo_to_ising_coefficients(q)
        assert abs(s-max([abs(v) for v in h.values()]+[abs(v) for v in j.values()]+[0]))<1e-12
        qdict={(i,j):q[i,j] for i in range(23) for j in range(i,23)}
        train_ids=base['initial_dataset']['candidate_ids']
        record={'seed':seed,'status':'running','initial_sha256':base['initial_sha256'],
          'fixed_model_sha256':base['fixed_model_sha256'],'source_pilot_sha256':sha(PILOT/f'json/seed_{seed}.json'),
          'checkpoint_sha256':base['checkpoint_sha256'],'base_Ising_scale_S':s,
          'protocol_sha256':sha(p),'code_sha256':hashes,'conditions':[]}
        save(target,record)
        try:
            for alpha in protocol['alphas']:
                bqm=build_penalty_bqm(qdict,[16,3,4],s*alpha,offset)
                runs=[]
                for rep in range(protocol['sampling_repetitions']):
                    pools=[];batch_info=[];t=time.perf_counter()
                    for batch in range(4):
                        sampler_seed=seed*100+rep*100000+batch
                        samples=neal.SimulatedAnnealingSampler().sample(bqm,num_reads=250,num_sweeps=1000,
                          seed=sampler_seed,beta_schedule_type='geometric')
                        arr=samples.record.sample[:,[list(samples.variables).index(i) for i in range(23)]]
                        pools.append(np.repeat(arr,samples.record.num_occurrences,axis=0).astype(np.int8))
                        batch_info.append({'seed':sampler_seed,'info':{k:v for k,v in samples.info.items() if k!='timing'}})
                    seconds=time.perf_counter()-t
                    bits=np.concatenate(pools);assert bits.shape==(1000,23)
                    result=measure(bits,q,offset,patterns,train_ids,truth,categories)
                    result.update({'repetition':rep,'generation_seconds':seconds,'sampler_batches':batch_info})
                    assert result['raw_invalid_count']+result['initial_duplicate_count']+result['within_unseen_pool_duplicate_count']+result['unique_unseen_candidates']==1000
                    runs.append(result)
                record['conditions'].append({'alpha':alpha,'lambda_internal':s*alpha,'runs':runs})
                save(target,record)
                print(f'seed={seed} alpha={alpha} feasible={np.mean([r["empirical_raw_feasible_rate"] for r in runs]):.4f}',flush=True)
            record['status']='completed';save(target,record)
        except Exception:
            record.update({'status':'failed','failure':traceback.format_exc()});save(target,record);failures.append(seed)
    if failures:raise RuntimeError(f'failed seeds retained {failures}')

if __name__=='__main__':main()
