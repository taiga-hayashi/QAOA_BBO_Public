"""Ideal full-state measurements at frozen pilot angles, plus fresh 10-read SA."""
import sys,json,time,traceback,platform,importlib.metadata
from pathlib import Path
import numpy as np
import neal
OUT=Path(__file__).resolve().parents[1];PILOT=OUT.parent/'fixed_fm_pilot'
sys.path.insert(0,str(PILOT/'py'))
from run_pilot import (PerovskitesEvaluator,score_candidates,sha,save,matrix,build_penalty_bqm,
 OpenQARPXYQAOA,OpenQARPStandardQAOA,_build_qubo_energies,AdaptivePenaltyTracker,REPO)

def draw(cdf,uniforms):
    return np.searchsorted(cdf,uniforms,side='right').astype(np.uint64)

def score(indices,q,offset,cats,patterns,initial,truth,pred):
    bits=((indices[:,None]>>np.arange(23,dtype=np.uint64))&1).astype(np.int8)
    r=score_candidates(bits,{},q,offset,cats,patterns,initial,truth,{'accepted_candidates':1})
    r['allowed_total_bb_budget']=21
    selected=r['accepted_candidates'][0] if r['accepted_count'] else None
    unseen_min=min(pred[i] for i in range(192) if i not in set(initial))
    r.update({'unseen_success':int(selected is not None),
      'fm_minimum_hit':int(selected is not None and abs(selected['fm_prediction']-unseen_min)<1e-10),
      'selected_fm_gap':selected['surrogate_gap_to_global_feasible_fm_min'] if selected else None,
      'selected_true_gap':selected['hse_gap'] if selected else None})
    assert r['accepted_count']<=1 and r['bb_evaluations_initial']+r['bb_evaluations_new']<=21
    assert r['raw_invalid_count']+r['initial_duplicate_count']+r['within_unseen_pool_duplicate_count']+r['unique_unseen_candidates']==len(indices)
    return r

def main():
    protocol=json.loads((OUT/'json/protocol.json').read_text());assert protocol['status']=='ready'
    validation=json.loads((OUT/'json/prevalidation.json').read_text())
    assert validation['status']=='passed' and validation['runner_sha256']==sha(Path(__file__))
    assert json.loads((PILOT/'json/artifact_verification.json').read_text())['status']=='passed'
    bb=PerovskitesEvaluator();cats=[list(c) for c in bb.candidates()];patterns=np.array([bb.encode(c) for c in cats],dtype=np.int8)
    truth=np.array([bb.evaluate(c) for c in cats]);valid=patterns.astype(np.uint64)@(1<<np.arange(23,dtype=np.uint64))
    code=[Path(__file__),PILOT/'py/run_pilot.py',REPO/'src/qarp_backend.py',REPO/'src/fmqa_solver.py',REPO/'intern_0924/src/penalty.py',REPO/'intern_0924/src/solvers.py']
    save(OUT/'json/environment.json',{'python':sys.version,'platform':platform.platform(),'executable':sys.executable,
      'versions':{k:importlib.metadata.version(k) for k in ['numpy','torch','openqarp','dwave-neal','scipy']},
      'code_sha256':{str(p.relative_to(REPO)):sha(p) for p in code},'protocol_sha256':sha(OUT/'json/protocol.json')})
    failures=[]
    for seed in protocol['seeds']:
        target=OUT/f'json/seed_{seed}.json'
        if target.exists():raise RuntimeError('preserve existing records, use new folder for rerun')
        base=json.loads((PILOT/f'json/seed_{seed}.json').read_text());assert base['status']=='completed'
        assert sha(PILOT/base['checkpoint'])==base['checkpoint_sha256']
        q=np.array(base['qubo']);offset=base['offset'];s=base['base_Ising_scale_S'];divisor=s if s>0 else 1.
        initial=base['initial_dataset']['candidate_ids'];pred=np.einsum('bi,ij,bj->b',patterns,q,patterns)+offset
        record={'seed':seed,'status':'running','initial_sha256':base['initial_sha256'],'fixed_model_sha256':base['fixed_model_sha256'],
          'source_sha256':sha(PILOT/f'json/seed_{seed}.json'),'checkpoint_sha256':base['checkpoint_sha256'],
          'protocol_sha256':sha(OUT/'json/protocol.json'),'qaoa':{},'sa10':{}}
        save(target,record)
        try:
            for method in protocol['methods']:
                old=base['methods'][method];gamma,beta=old['angles'];xy=method=='XY-FMQAOA';alpha=0 if xy else 5
                backend=OpenQARPXYQAOA(q/divisor,[16,3,4]) if xy else OpenQARPStandardQAOA(q/divisor,[16,3,4],s*alpha/divisor)
                start=time.perf_counter();state=backend.statevector([gamma],[beta]);prob=np.abs(state)**2;norm=float(prob.sum())
                assert abs(norm-1)<1e-10;prob/=norm;del state,backend
                feasible_prob=prob[valid].copy();np.testing.assert_allclose(feasible_prob,old['feasible_basis_probabilities'],atol=1e-12,rtol=1e-10)
                if xy:assert abs(feasible_prob.sum()-1)<1e-10 and alpha==0
                qdict={(i,j):q[i,j]/divisor for i in range(23) for j in range(i,23)}
                bqm=build_penalty_bqm(qdict,[16,3,4],s*alpha/divisor,offset/divisor);qd,bias=bqm.to_qubo()
                energies=_build_qubo_energies(matrix(qd,23),bias,np.dtype('float64'))
                mean=0.;second=0.
                for lo in range(0,len(prob),65536):
                    pr=prob[lo:lo+65536];e=energies[lo:lo+65536];mean+=float(pr@e);second+=float(pr@(e*e))
                variance=max(0.,second-mean*mean)
                expected=next(v['expected_normalized_surrogate_plus_penalty'] for v in old['angle_search'] if v['gamma']==gamma and v['beta']==beta)
                assert abs(mean-expected)<1e-9
                cdf=np.cumsum(prob);cdf[-1]=1.;del prob
                setup_seconds=time.perf_counter()-start
                unseen=np.ones(192,bool);unseen[initial]=False;pnew=float(feasible_prob[unseen].sum())
                minpred=pred[unseen].min();popt=float(feasible_prob[unseen & (np.abs(pred-minpred)<1e-10)].sum())
                conditions={str(n):[] for n in protocol['shot_counts']}
                for rep in range(protocol['measurement_repetitions']):
                    uniforms=np.random.default_rng(np.random.SeedSequence([seed,rep,1009,73])).random(1000)
                    allindices=draw(cdf,uniforms);assert np.all(allindices<1<<23)
                    for n in protocol['shot_counts']:
                        indices=allindices[:n];r=score(indices,q,offset,cats,patterns,initial,truth,pred)
                        # Direct cost formula validates measured energies including invalid samples.
                        bits=((indices[:,None]>>np.arange(23,dtype=np.uint64))&1).astype(np.int8)
                        violations=sum((bits[:,a:b].sum(axis=1)-1)**2 for a,b in [(0,16),(16,19),(19,23)])
                        direct=(np.einsum('bi,ij,bj->b',bits,q,bits)+offset+s*alpha*violations)/divisor
                        np.testing.assert_allclose(energies[indices],direct,atol=1e-9,rtol=1e-12)
                        estimate=float(energies[indices].mean())
                        r.update({'repetition':rep,'shots':n,'energy_estimate':estimate,'energy_error':estimate-mean,
                          'feasible_rate_abs_error':abs(r['empirical_raw_feasible_rate']-float(feasible_prob.sum()))})
                        conditions[str(n)].append(r)
                record['qaoa'][method]={'angles':[gamma,beta],'alpha':alpha,'lambda_internal':s*alpha,'normalization_divisor':divisor,
                  'raw_norm':norm,'exact_feasible_probability':float(feasible_prob.sum()),'exact_unseen_probability':pnew,
                  'exact_fm_minimum_probability':popt,'exact_energy_mean':mean,'exact_energy_variance':variance,
                  'selected_circuit_setup_seconds':setup_seconds,'conditions':conditions}
                del energies,cdf;save(target,record)
                print(f'seed={seed} {method} exact feasible={float(feasible_prob.sum()):.8g}',flush=True)
            qdict={(i,j):q[i,j] for i in range(23) for j in range(i,23)}
            for method in ['Adaptive-FMQA','LargePenalty-FMQA']:
                runs=[]
                for rep in range(protocol['measurement_repetitions']):
                    tracker=AdaptivePenaltyTracker();pool=[];batches=[];start=time.perf_counter()
                    for batch,reads in enumerate([3,3,2,2]):
                        alpha=tracker.get_current_alpha() if method=='Adaptive-FMQA' else 100
                        samples=neal.SimulatedAnnealingSampler().sample(build_penalty_bqm(qdict,[16,3,4],s*alpha,offset),
                          num_reads=reads,num_sweeps=1000,seed=seed*100+rep*100000+batch,beta_schedule_type='geometric')
                        arr=samples.record.sample[:,[list(samples.variables).index(i) for i in range(23)]]
                        bits=np.repeat(arr,samples.record.num_occurrences,axis=0).astype(np.int8)
                        rate=float(np.mean((bits[:,:16].sum(axis=1)==1)&(bits[:,16:19].sum(axis=1)==1)&(bits[:,19:].sum(axis=1)==1)))
                        nxt=tracker.update(rate) if method=='Adaptive-FMQA' else alpha
                        batches.append({'reads':reads,'alpha':alpha,'lambda_internal':s*alpha,'raw_feasible_rate':rate,'next_alpha':nxt,
                          'neal_info':{k:v for k,v in samples.info.items() if k!='timing'}});pool.append(bits)
                    seconds=time.perf_counter()-start;bits=np.concatenate(pool);assert len(bits)==10
                    indices=bits.astype(np.uint64)@(1<<np.arange(23,dtype=np.uint64))
                    r=score(indices,q,offset,cats,patterns,initial,truth,pred)
                    r.update({'repetition':rep,'sa_generation_seconds':seconds,'batches':batches});runs.append(r)
                record['sa10'][method]=runs;save(target,record)
            record['status']='completed';save(target,record)
        except Exception:
            record.update({'status':'failed','failure':traceback.format_exc()});save(target,record);failures.append(seed)
            print(record['failure'],flush=True)
    if failures:raise RuntimeError(f'failed seeds retained: {failures}')

if __name__=='__main__':main()
