"""Standard p1 QAOA: identical angle grid per alpha, then ideal finite measurements."""
import sys,json,time,traceback,platform,argparse,concurrent.futures
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parents[1];PROBLEM=OUT.parent;PILOT=PROBLEM/'fixed_fm_pilot';SHOTS=PROBLEM/'qaoa_shot_accuracy'
sys.path[:0]=[str(SHOTS/'py'),str(PILOT/'py')]
from run_shots import draw,score
from run_pilot import (PerovskitesEvaluator,sha,save,qaoa_generate,matrix,build_penalty_bqm,
 OpenQARPStandardQAOA,qubo_to_ising_coefficients,_build_qubo_energies,REPO)

def run_seed(seed):
    protocol=json.loads((OUT/'json/protocol.json').read_text());target=OUT/f'json/seed_{seed}.json'
    if target.exists():raise RuntimeError('existing records preserved; rerun in a new folder')
    bb=PerovskitesEvaluator();cats=[list(c) for c in bb.candidates()];patterns=np.array([bb.encode(c) for c in cats],dtype=np.int8)
    truth=np.array([bb.evaluate(c) for c in cats]);valid=patterns.astype(np.uint64)@(1<<np.arange(23,dtype=np.uint64))
    base=json.loads((PILOT/f'json/seed_{seed}.json').read_text());assert base['status']=='completed'
    assert sha(PILOT/base['checkpoint'])==base['checkpoint_sha256']
    q=np.array(base['qubo']);offset=base['offset'];s=base['base_Ising_scale_S'];divisor=s if s>0 else 1.
    initial=base['initial_dataset']['candidate_ids'];pred=np.einsum('bi,ij,bj->b',patterns,q,patterns)+offset
    record={'seed':seed,'status':'running','source_sha256':sha(PILOT/f'json/seed_{seed}.json'),
      'initial_sha256':base['initial_sha256'],'fixed_model_sha256':base['fixed_model_sha256'],
      'checkpoint_sha256':base['checkpoint_sha256'],'protocol_sha256':sha(OUT/'json/protocol.json'),'conditions':[]}
    save(target,record)
    try:
        base_q=q/divisor;bias=offset/divisor
        base_energy=_build_qubo_energies(base_q,bias,np.dtype('float64'))
        unseen=np.ones(192,bool);unseen[initial]=False;minpred=pred[unseen].min()
        for alpha in protocol['alphas']:
            started=time.perf_counter()
            if alpha==5:
                old=base['methods']['Penalty-FMQAOA'];meta={k:old[k] for k in ['angles','angle_search','feasible_basis_probabilities']}
                search_origin='reused validated original identical alpha5 search'
            else:
                settings={'penalty_qaoa_alpha':alpha,'qaoa':{**protocol['qaoa'],'shots':10}}
                _,meta=qaoa_generate(q,offset,s,[16,3,4],seed,False,settings,valid)
                search_origin='new exact expectation nine-point search'
            gamma,beta=meta['angles'];backend=OpenQARPStandardQAOA(base_q,[16,3,4],s*alpha/divisor)
            state=backend.statevector([gamma],[beta]);prob=np.abs(state)**2;norm=float(prob.sum());assert abs(norm-1)<1e-10
            prob/=norm;del state,backend
            feasible_prob=prob[valid].copy();np.testing.assert_allclose(feasible_prob,meta['feasible_basis_probabilities'],atol=1e-12,rtol=1e-9)
            qdict={(i,j):base_q[i,j] for i in range(23) for j in range(i,23)}
            bqm=build_penalty_bqm(qdict,[16,3,4],s*alpha/divisor,bias);qd,b=bqm.to_qubo();total_q=matrix(qd,23)
            energies=_build_qubo_energies(total_q,b,np.dtype('float64'))
            h,j=qubo_to_ising_coefficients(total_q);cost_scale=max([abs(v) for v in h.values()]+[abs(v) for v in j.values()]+[1e-15])
            mean=second=base_mean=base_second=0.
            for lo in range(0,len(prob),65536):
                p=prob[lo:lo+65536];e=energies[lo:lo+65536];f=base_energy[lo:lo+65536]
                mean+=float(p@e);second+=float(p@(e*e));base_mean+=float(p@f);base_second+=float(p@(f*f))
            expected=next(v['expected_normalized_surrogate_plus_penalty'] for v in meta['angle_search'] if v['gamma']==gamma and v['beta']==beta)
            assert abs(mean-expected)<1e-6
            variance=max(0.,second-mean*mean);base_variance=max(0.,base_second-base_mean*base_mean)
            cdf=np.cumsum(prob);cdf[-1]=1.;del prob
            pnew=float(feasible_prob[unseen].sum());popt=float(feasible_prob[unseen & (np.abs(pred-minpred)<1e-10)].sum())
            fm_top=set(np.argsort(pred,kind='stable')[:10].tolist());true_top=set(np.argsort(truth,kind='stable')[:10].tolist())
            runs={str(n):[] for n in protocol['shot_counts']}
            for rep in range(protocol['measurement_repetitions']):
                uniforms=np.random.default_rng(np.random.SeedSequence([seed,rep,1009,73])).random(1000);allindices=draw(cdf,uniforms)
                for n in protocol['shot_counts']:
                    indices=allindices[:n];r=score(indices,q,offset,cats,patterns,initial,truth,pred)
                    bits=((indices[:,None]>>np.arange(23,dtype=np.uint64))&1).astype(np.int8)
                    violations=sum((bits[:,a:z].sum(axis=1)-1)**2 for a,z in [(0,16),(16,19),(19,23)])
                    direct=(np.einsum('bi,ij,bj->b',bits,q,bits)+offset+s*alpha*violations)/divisor
                    np.testing.assert_allclose(energies[indices],direct,atol=1e-7,rtol=1e-12)
                    r.update({'repetition':rep,'shots':n,'energy_estimate':float(energies[indices].mean()),
                      'energy_error':float(energies[indices].mean()-mean),'base_FM_energy_error':float(base_energy[indices].mean()-base_mean),
                      'feasible_rate_abs_error':abs(r['empirical_raw_feasible_rate']-float(feasible_prob.sum()))})
                    runs[str(n)].append(r)
            if alpha==5:
                prior=json.loads((SHOTS/f'json/seed_{seed}.json').read_text())['qaoa']['Penalty-FMQAOA']
                for n in protocol['shot_counts']:
                    for r,ref in zip(runs[str(n)],prior['conditions'][str(n)]):
                        assert r['raw_basis_indices']==ref['raw_basis_indices']
                        assert abs(r['energy_error']-ref['energy_error'])<1e-8
            item={'alpha':alpha,'lambda_internal':s*alpha,'base_scale':s,'normalization_divisor':divisor,
              'angles':[gamma,beta],'angle_search':meta['angle_search'],'search_origin':search_origin,'raw_norm':norm,
              'exact_feasible_probability':float(feasible_prob.sum()),'exact_unseen_probability':pnew,'exact_fm_minimum_probability':popt,
              'exact_unseen_FM_top5pct_probability':float(sum(feasible_prob[i] for i in fm_top if unseen[i])),
              'exact_unseen_true_top5pct_probability':float(sum(feasible_prob[i] for i in true_top if unseen[i])),
              'exact_energy_mean':mean,'exact_energy_variance':variance,'base_FM_mean':base_mean,'base_FM_variance':base_variance,
              'penalized_normalized_Ising_scale':cost_scale,'search_and_sampling_seconds':time.perf_counter()-started,'shots':runs}
            record['conditions'].append(item);save(target,record)
            del energies,cdf
            print(f'FINISHED seed={seed} alpha={alpha} P_feasible={item["exact_feasible_probability"]:.7g}',flush=True)
        record['status']='completed';save(target,record);return seed
    except Exception:
        record.update({'status':'failed','failure':traceback.format_exc()});save(target,record);raise

def main():
    args=argparse.ArgumentParser();args.add_argument('--workers',type=int,default=3);opts=args.parse_args()
    protocol=json.loads((OUT/'json/protocol.json').read_text());assert protocol['status']=='ready'
    validation=json.loads((OUT/'json/prevalidation.json').read_text());assert validation['status']=='passed' and validation['runner_sha256']==sha(Path(__file__))
    assert json.loads((PILOT/'json/artifact_verification.json').read_text())['status']=='passed'
    save(OUT/'json/environment.json',{'python':sys.version,'platform':platform.platform(),'workers':opts.workers,
      'protocol_sha256':sha(OUT/'json/protocol.json'),'code_sha256':{str(p.relative_to(REPO)):sha(p) for p in [Path(__file__),SHOTS/'py/run_shots.py',PILOT/'py/run_pilot.py',REPO/'src/qarp_backend.py']}})
    failures=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=opts.workers) as executor:
        jobs={executor.submit(run_seed,seed):seed for seed in protocol['seeds']}
        for job in concurrent.futures.as_completed(jobs):
            try:print(f'COMPLETED seed={job.result()}',flush=True)
            except Exception:failures.append(jobs[job]);print(traceback.format_exc(),flush=True)
    if failures:raise RuntimeError(f'failed seeds retained {failures}')

if __name__=='__main__':main()
