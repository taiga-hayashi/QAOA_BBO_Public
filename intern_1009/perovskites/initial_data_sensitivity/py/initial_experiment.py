"""Initial-count ablation; canonical model, sampler and filter reused."""
import sys,json,time,traceback
from pathlib import Path
import numpy as np
import torch
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'bbo_sample_count_comparison/py'))
from experiment import generate,CFG,fit,select,PerovskitesEvaluator,save,sha,model_hash,AdaptivePenaltyTracker,REPO
P=json.loads((OUT/'json/protocol.json').read_text())
CODE=[Path(__file__),ROOT/'bbo_sample_count_comparison/py/experiment.py',ROOT/'fmqa_adamw_bbo/py/run_bbo.py',REPO/'src/fm.py',REPO/'src/fm_to_qubo.py',REPO/'src/qarp_backend.py',REPO/'src/fmqa_solver.py',REPO/'intern_0924/src/penalty.py',ROOT/'py/perovskites_evaluator.py']

def prior(method,seed):return ROOT/f'bbo_sample_count_comparison/json/{method}_n100_seed{seed}.json'

def prevalidate():
 assert P['status']=='ready';v=json.loads((ROOT/'bbo_sample_count_comparison/json/prevalidation.json').read_text());assert v['status']=='passed'
 for n,h in v['source_sha256'].items():assert sha(REPO/n)==h
 bb=PerovskitesEvaluator();assert bb.data_sha256==P['data_sha256'];cats=list(bb.candidates());x=np.array([bb.encode(c) for c in cats]);checks=[]
 for seed in P['seeds']:
  base=json.loads((ROOT/f'fixed_fm_pilot/json/seed_{seed}.json').read_text());full=base['initial_dataset']['candidate_ids']
  for n in P['initial_counts']:
   ids=full[:n];assert len(set(ids))==n and ids==full[:n];m,qd,q,b=fit(CFG,seed,x[ids],[bb.evaluate(cats[i]) for i in ids]);pred=np.einsum('bi,ij,bj->b',x,q,x)+b
   with torch.no_grad():err=float(np.max(np.abs(pred-m(torch.tensor(x,dtype=torch.float32)).numpy())))
   assert err<1e-4
   if n==20:
    for method in P['methods']:
     bits,_=generate(qd,q,b,pred,method,100,seed,1,AdaptivePenaltyTracker(),x);e=json.loads(prior(method,seed).read_text())['events'][0];assert bits.tolist()==e['raw_bits'] and model_hash(m)==e['model_sha256'] and select(bits,pred,x,set(ids))[0]==e['selected_id']
   checks.append({'seed':seed,'initial_count':n,'parity_max_error':err,'organic_coverage':int(np.sum(x[ids,:16].sum(axis=0)>0))})
 save(OUT/'json/prevalidation.json',{'status':'passed','initial_models':checks,'first_cycle20_15_reproduction':'passed','inherited_N6_N9_N23_full_circuit_parity':'passed,unchanged hashes','source_sha256':{str(c.relative_to(REPO)):sha(c) for c in CODE}});print('prevalidation passed',flush=True)

def run():
 audit=json.loads((OUT/'json/prevalidation.json').read_text());assert audit['source_sha256']=={str(c.relative_to(REPO)):sha(c) for c in CODE}
 bb=PerovskitesEvaluator();cats=list(bb.candidates());x=np.array([bb.encode(c) for c in cats],dtype=np.int8);env=json.loads((ROOT/'bbo_sample_count_comparison/json/environment.json').read_text());env.update({'source_sha256':audit['source_sha256'],'protocol_sha256':sha(OUT/'json/protocol.json')});save(OUT/'json/environment.json',env)
 for seed in P['seeds']:
  base=json.loads((ROOT/f'fixed_fm_pilot/json/seed_{seed}.json').read_text())
  for n in P['initial_counts']:
   for method in P['methods']:
    name=f'{method}_initial{n}_seed{seed}';path=OUT/f'json/{name}.json';assert not path.exists();ids=list(base['initial_dataset']['candidate_ids'][:n]);vals=[float(bb.evaluate(cats[i])) for i in ids];states=[];tracker=AdaptivePenaltyTracker();cycles=80-n;d={'status':'running','method':method,'seed':seed,'initial_count':n,'cycles_planned':cycles,'initial_ids':list(ids),'initial_values':list(vals),'source20_initial_sha256':base['initial_sha256'],'protocol_sha256':sha(OUT/'json/protocol.json'),'events':[]};start=time.perf_counter();save(path,d)
    try:
     for cycle in range(1,cycles+1):
      t=time.perf_counter();m,qd,q,b=fit(CFG,seed+100000*(cycle-1),x[ids],vals);pred=np.einsum('bi,ij,bj->b',x,q,x)+b;trainsecs=time.perf_counter()-t
      with torch.no_grad():err=float(np.max(np.abs(pred-m(torch.tensor(x,dtype=torch.float32)).numpy())))
      assert err<1e-4 and np.isfinite(pred).all();t=time.perf_counter();bits,meta=generate(qd,q,b,pred,method,100,seed,cycle,tracker,x);assert len(bits)==100;chosen,filters=select(bits,pred,x,set(ids));gen=time.perf_counter()-t;unseen=[i for i in range(192) if i not in ids];umin=float(pred[unseen].min());nextalpha=tracker.update(filters['raw_feasible_rate']) if method=='Adaptive-FMQA' else meta['alpha']
      e={'cycle':cycle,'train_ids_before':list(ids),'evaluations_before':len(ids),'model_seed':seed+100000*(cycle-1),'model_sha256':model_hash(m),'fm_predictions':pred.tolist(),'raw_bits':bits.tolist(),'selected_id':chosen,'selected_prediction':float(pred[chosen]) if chosen is not None else None,'selected_value':None,'unseen_fm_min':umin,'selected_unseen_fm_gap':float(pred[chosen]-umin) if chosen is not None else None,'next_alpha':nextalpha,'parity_max_error':err,'train_seconds':trainsecs,'generation_seconds':gen,**meta,**filters};states.append({'cycle':cycle,'state_dict':m.state_dict(),'model_sha256':model_hash(m)})
      if chosen is not None:assert chosen not in ids and len(ids)<80;val=float(bb.evaluate(cats[chosen]));ids.append(chosen);vals.append(val);e['selected_value']=val
      e.update({'evaluations_after':len(ids),'best_so_far':min(vals)});assert len(ids)==len(set(ids))<=80;d['events'].append(e);save(path,d)
     cp=OUT/f'data/checkpoints/{name}.pt';torch.save(states,cp);d.update({'status':'completed','final_ids':ids,'final_values':vals,'actual_evaluations':len(ids),'final_best':min(vals),'checkpoint':str(cp.relative_to(OUT)),'checkpoint_sha256':sha(cp),'total_seconds':time.perf_counter()-start});save(path,d);print(name,'best',min(vals),'evals',len(ids),flush=True)
    except Exception:d.update({'status':'failed','traceback':traceback.format_exc()});save(path,d);raise

if __name__=='__main__':globals()[sys.argv[1]]()
