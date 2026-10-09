"""Count ablation: canonical models/samplers, same frozen loop and budgets."""
import sys,json,time,traceback
from pathlib import Path
import numpy as np
import torch
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'fmqa_adamw_bbo/py'))
from run_bbo import fit,select,PerovskitesEvaluator,save,sha,model_hash,matrix,build_penalty_bqm,qubo_to_ising_coefficients,AdaptivePenaltyTracker,REPO,neal
from qarp_backend import OpenQARPCompactXYQAOA
P=json.loads((OUT/'json/protocol.json').read_text());CFG={'id':'proposed','rank':1,'optimizer':'AdamW','weight_decay':.01}
CODE=[Path(__file__),REPO/'src/fm.py',REPO/'src/fm_to_qubo.py',REPO/'src/fmqa_solver.py',REPO/'src/qarp_backend.py',ROOT/'fmqa_adamw_bbo/py/run_bbo.py',REPO/'intern_0924/src/penalty.py',ROOT/'py/perovskites_evaluator.py']

def baseline(method,seed):
 return ROOT/f'fmxy_adamw_bbo/json/seed_{seed}.json' if method=='XY-FMQAOA' else ROOT/f'fmqa_adamw_bbo/json/proposed_{method}_seed{seed}.json'

def generate(qd,q,b,pred,method,count,seed,cycle,tracker,x):
 h,j=qubo_to_ising_coefficients(q);s=max([abs(v) for v in h.values()]+[abs(v) for v in j.values()]+[0.]);rngseed=seed*100000+cycle*1000
 if method=='XY-FMQAOA':
  div=s if s>0 else 1.;backend=OpenQARPCompactXYQAOA(q/div,[16,3,4]);best=None;search=[]
  for gamma in P['xy']['gammas']:
   for beta in P['xy']['betas']:
    prob=backend.feasible_probabilities([gamma],[beta]);value=float(prob@(pred/div));search.append({'gamma':gamma,'beta':beta,'expected_normalized_FM':value,'feasible_mass':float(prob.sum())})
    if best is None or value<best[0]:best=(value,gamma,beta,prob.copy())
  order=np.argsort(x.astype(np.uint64)@(1<<np.arange(23,dtype=np.uint64)));cdf=np.cumsum(best[3][order]);u=np.random.default_rng(rngseed).random(count)*cdf[-1];indices=order[np.searchsorted(cdf,u,side='left')]
  return x[indices],{'alpha':0.,'lambda':0.,'base_Ising_scale_S':s,'normalization_divisor':div,'angle_search':search,'chosen_angles':[best[1],best[2]],'feasible_probabilities':best[3].tolist(),'raw_candidate_ids':indices.tolist(),'sampler_seed':rngseed}
 alpha=tracker.get_current_alpha() if method=='Adaptive-FMQA' else 100.;lam=s*alpha;bqm=build_penalty_bqm(qd,[16,3,4],lam,b);ss=neal.SimulatedAnnealingSampler().sample(bqm,num_reads=count,num_sweeps=1000,beta_schedule_type='geometric',seed=rngseed)
 arr=ss.record.sample[:,[list(ss.variables).index(i) for i in range(23)]];bits=np.repeat(arr,ss.record.num_occurrences,axis=0).astype(np.int8)
 return bits,{'alpha':alpha,'lambda':lam,'base_Ising_scale_S':s,'sampler_seed':rngseed,'neal_info':{k:v for k,v in ss.info.items() if k!='timing'}}

def prevalidate():
 assert P['status']=='ready';old=json.loads((ROOT/'fmxy_adamw_bbo/json/prevalidation.json').read_text());assert old['status']=='passed'
 for name,h in old['source_sha256'].items():assert sha(REPO/name)==h
 assert (REPO/'src/qarp_backend.py').read_bytes()==(REPO/'intern/src/qarp_backend.py').read_bytes()
 bb=PerovskitesEvaluator();assert bb.data_sha256=='ba55731336cf68a2fcc2a8c6b542ee06d4c238c91ced27d243ce9de95f8d6518';cats=list(bb.candidates());x=np.array([bb.encode(c) for c in cats]);checks=[]
 for seed in P['seeds']:
  init=json.loads((ROOT/f'fixed_fm_pilot/json/seed_{seed}.json').read_text());ids=init['initial_dataset']['candidate_ids'];m,qd,q,b=fit(CFG,seed,x[ids],[bb.evaluate(cats[i]) for i in ids]);pred=np.einsum('bi,ij,bj->b',x,q,x)+b
  for method in P['methods']:
   previous=json.loads(baseline(method,seed).read_text());e=previous['events'][0]
   for count in P['counts']:
    bits,meta=generate(qd,q,b,pred,method,count,seed,1,AdaptivePenaltyTracker(),x);assert len(bits)==count;chosen,_=select(bits,pred,x,set(ids))
    if count==10:assert bits.tolist()==e['raw_bits'] and chosen==e['selected_id'] and model_hash(m)==e['model_sha256'];np.testing.assert_allclose(pred,e['fm_predictions'],atol=0,rtol=0)
   checks.append({'seed':seed,'method':method,'first_cycle_10_reproduction':'exact','count100_check':'passed'})
 save(OUT/'json/prevalidation.json',{'status':'passed','checks':checks,'inherited_N6_N9_N23_full_OpenQARP_parity':'passed; shared hashes unchanged','source_sha256':{str(c.relative_to(REPO)):sha(c) for c in CODE}});print('prevalidated15 baselines and100 output checks',flush=True)

def run():
 audit=json.loads((OUT/'json/prevalidation.json').read_text());assert audit['source_sha256']=={str(c.relative_to(REPO)):sha(c) for c in CODE}
 bb=PerovskitesEvaluator();cats=list(bb.candidates());x=np.array([bb.encode(c) for c in cats],dtype=np.int8);env=json.loads((ROOT/'fmxy_adamw_bbo/json/environment.json').read_text());env.update({'source_sha256':audit['source_sha256'],'protocol_sha256':sha(OUT/'json/protocol.json')});save(OUT/'json/environment.json',env)
 for seed in P['seeds']:
  initial=json.loads((ROOT/f'fixed_fm_pilot/json/seed_{seed}.json').read_text())
  for method in P['methods']:
   for count in P['counts']:
    name=f'{method}_n{count}_seed{seed}';path=OUT/f'json/{name}.json';assert not path.exists();ids=list(initial['initial_dataset']['candidate_ids']);vals=[float(bb.evaluate(cats[i])) for i in ids];tracker=AdaptivePenaltyTracker();states=[];d={'status':'running','seed':seed,'method':method,'count':count,'initial_sha256':initial['initial_sha256'],'initial_ids':list(ids),'initial_values':list(vals),'protocol_sha256':sha(OUT/'json/protocol.json'),'events':[]};start=time.perf_counter();save(path,d)
    try:
     for cycle in range(1,61):
      t=time.perf_counter();m,qd,q,b=fit(CFG,seed+100000*(cycle-1),x[ids],vals);pred=np.einsum('bi,ij,bj->b',x,q,x)+b;trainsecs=time.perf_counter()-t
      with torch.no_grad():err=float(np.max(np.abs(pred-m(torch.tensor(x,dtype=torch.float32)).numpy())))
      assert err<1e-4 and np.isfinite(pred).all();t=time.perf_counter();bits,meta=generate(qd,q,b,pred,method,count,seed,cycle,tracker,x);assert len(bits)==count;chosen,filters=select(bits,pred,x,set(ids));gen=time.perf_counter()-t
      unseen=[i for i in range(192) if i not in ids];unseenmin=float(pred[unseen].min());globalmin=float(pred.min());nextalpha=tracker.update(filters['raw_feasible_rate']) if method=='Adaptive-FMQA' else meta['alpha']
      e={'cycle':cycle,'train_ids_before':list(ids),'evaluations_before':len(ids),'model_seed':seed+100000*(cycle-1),'model_sha256':model_hash(m),'fm_predictions':pred.tolist(),'raw_bits':bits.tolist(),'selected_id':chosen,'selected_prediction':float(pred[chosen]) if chosen is not None else None,'selected_value':None,'unseen_fm_min':unseenmin,'global_feasible_fm_min':globalmin,'selected_unseen_fm_gap':float(pred[chosen]-unseenmin) if chosen is not None else None,'selected_global_fm_gap':float(pred[chosen]-globalmin) if chosen is not None else None,'next_alpha':nextalpha,'train_seconds':trainsecs,'generation_seconds':gen,'parity_max_error':err,**meta,**filters};states.append({'cycle':cycle,'state_dict':m.state_dict(),'model_sha256':model_hash(m)})
      if chosen is not None:assert chosen not in ids and len(ids)<80;val=float(bb.evaluate(cats[chosen]));ids.append(chosen);vals.append(val);e['selected_value']=val
      assert len(ids)==len(set(ids))<=80;e.update({'evaluations_after':len(ids),'best_so_far':min(vals)});d['events'].append(e);save(path,d)
     cp=OUT/f'data/checkpoints/{name}.pt';torch.save(states,cp);d.update({'status':'completed','final_ids':ids,'final_values':vals,'actual_evaluations':len(ids),'final_best':min(vals),'checkpoint':str(cp.relative_to(OUT)),'checkpoint_sha256':sha(cp),'total_seconds':time.perf_counter()-start});save(path,d);print(name,'best',min(vals),'evals',len(ids),flush=True)
    except Exception:d.update({'status':'failed','traceback':traceback.format_exc()});save(path,d);raise

if __name__=='__main__':globals()[sys.argv[1]]()
