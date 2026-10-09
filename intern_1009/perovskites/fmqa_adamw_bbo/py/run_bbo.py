"""Closed-loop lookup FMQA pilot; canonical FM, penalty builder and tracker."""
import sys,json,time,traceback,platform,importlib.metadata
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wilcoxon
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'fixed_fm_pilot/py'))
from run_pilot import PerovskitesEvaluator,train_factorization_machine,fm_to_qubo,matrix,sha,save,model_hash,build_penalty_bqm,qubo_to_ising_coefficients,AdaptivePenaltyTracker,REPO,neal
P=json.loads((OUT/'json/protocol.json').read_text())
CODE=[Path(__file__),REPO/'src/fm.py',REPO/'src/fm_to_qubo.py',REPO/'src/fmqa_solver.py',REPO/'intern_0924/src/penalty.py',ROOT/'py/perovskites_evaluator.py']

def select(bits,pred,patterns,seen):
 lookup={tuple(row):i for i,row in enumerate(patterns)};ids=[lookup.get(tuple(row)) for row in bits];unseen={i for i in ids if i is not None and i not in seen}
 chosen=min(unseen,key=lambda i:(pred[i],tuple(patterns[i]))) if unseen else None
 return chosen,{'raw_feasible_rate':sum(i is not None for i in ids)/len(ids),'invalid_count':sum(i is None for i in ids),'evaluated_duplicate_count':sum(i in seen for i in ids if i is not None),'unique_unseen_count':len(unseen),'within_unseen_duplicate_count':sum(i is not None and i not in seen for i in ids)-len(unseen)}

def fit(cfg,s,x,y):
 torch.set_num_threads(1);torch.manual_seed(s);torch.use_deterministic_algorithms(True)
 m=train_factorization_machine(torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.float32),23,k=cfg['rank'],epochs=P['training']['epochs'],learning_rate=P['training']['learning_rate'],optimizer_name=cfg['optimizer'],weight_decay=cfg['weight_decay']);m.eval();qd,b=fm_to_qubo(m);q=matrix(qd,23);return m,qd,q,b

def prevalidate():
 assert P['status']=='ready';bb=PerovskitesEvaluator();assert bb.data_sha256==P['data_sha256'];cats=list(bb.candidates());x=np.array([bb.encode(c) for c in cats]);assert x.shape==(192,23)
 manifest=json.loads((ROOT/'json/problem_manifest.json').read_text());assert manifest['prevalidation']['legacy_n6_n9_passed'] and manifest['prevalidation']['target_group_sizes_passed']
 oldchecks=json.loads((ROOT/'fm_regularization_diagnostic/json/delivery_verification.json').read_text());assert oldchecks['status']=='passed'
 # Pure selection tests, including invalid bits, evaluated duplicate, pool duplicate, tie and no refill.
 pred=np.arange(192,dtype=float);bits=np.array([x[0],x[1],x[1],np.zeros(23),x[2]])
 chosen,meta=select(bits,pred,x,{0});assert chosen==1 and meta=={'raw_feasible_rate':.8,'invalid_count':1,'evaluated_duplicate_count':1,'unique_unseen_count':2,'within_unseen_duplicate_count':1}
 assert select(np.array([x[0],np.zeros(23)]),pred,x,{0})[0] is None
 pred[:]=1.;assert select(np.array([x[1],x[2]]),pred,x,set())[0]==min([1,2],key=lambda i:tuple(x[i]))
 errors=[]
 for s in P['seeds']:
  base=json.loads((ROOT/f'fixed_fm_pilot/json/seed_{s}.json').read_text());tr=base['initial_dataset']['candidate_ids'];y=[bb.evaluate(cats[i]) for i in tr];old=json.loads((ROOT/f'fm_regularization_diagnostic/json/seed_{s}.json').read_text())
  for cfg in P['fm_settings']:
   m,qd,q,b=fit(cfg,s,x[tr],y);pred=np.einsum('bi,ij,bj->b',x,q,x)+b
   ident=f"k{cfg['rank']}_{cfg['optimizer']}_wd{cfg['weight_decay']}";f=next(f for f in old['fits'] if f['config']['id']==ident);err=float(np.max(np.abs(pred-np.array(f['predictions']))));assert err<1e-6;errors.append(err)
   h,j=qubo_to_ising_coefficients(q);scale=max([abs(v) for v in h.values()]+[abs(v) for v in j.values()]+[0.]);lam=100*scale;bqm=build_penalty_bqm(qd,[16,3,4],lam,b)
   checks=np.concatenate([x,np.random.default_rng(s).integers(0,2,(100,23))]);baseenergy=np.einsum('bi,ij,bj->b',checks,q,checks)+b
   violation=sum((checks[:,a:z].sum(axis=1)-1)**2 for a,z in [(0,16),(16,19),(19,23)])
   np.testing.assert_allclose(bqm.energies((checks,list(range(23)))),baseenergy+lam*violation,atol=1e-8,rtol=1e-10)
 save(OUT/'json/prevalidation.json',{'status':'passed','code_sha256':{str(c.relative_to(REPO)):sha(c) for c in CODE},'first_cycle_10_reproduction_errors':errors,'penalty_and_selection_tests':'passed','N6_N9_N23':'inherited passed; no circuit changes'})
 print('prevalidation passed',flush=True)

def run():
 v=json.loads((OUT/'json/prevalidation.json').read_text());assert v['code_sha256']=={str(c.relative_to(REPO)):sha(c) for c in CODE}
 bb=PerovskitesEvaluator();assert bb.data_sha256==P['data_sha256'];cats=[list(c) for c in bb.candidates()];x=np.array([bb.encode(c) for c in cats],dtype=np.int8)
 save(OUT/'json/environment.json',{'python':sys.version,'platform':platform.platform(),'architecture':platform.machine(),'device':'CPU','threads':1,'memory_capacity':'unmeasured','versions':{n:importlib.metadata.version(n) for n in ['numpy','torch','scipy','matplotlib','dwave-neal','dimod']},'code_sha256':v['code_sha256'],'protocol_sha256':sha(OUT/'json/protocol.json')})
 for seed in P['seeds']:
  basepath=ROOT/f'fixed_fm_pilot/json/seed_{seed}.json';base=json.loads(basepath.read_text())
  for cfg in P['fm_settings']:
   for method in P['methods']:
    name=f"{cfg['id']}_{method}_seed{seed}";target=OUT/f'json/{name}.json';assert not target.exists(),'preserve completed or partial results'
    ids=list(base['initial_dataset']['candidate_ids']);ys=[float(bb.evaluate(cats[i])) for i in ids];initialys=list(ys);seen=set(ids);calls=len(ids);tracker=AdaptivePenaltyTracker(initial_alpha=1.,min_alpha=.01,max_alpha=100.)
    d={'status':'running','seed':seed,'fm_setting':cfg,'method':method,'source_initial_sha256':base['initial_sha256'],'source_file_sha256':sha(basepath),'initial_ids':list(ids),'initial_values':initialys,'protocol_sha256':sha(OUT/'json/protocol.json'),'events':[]};save(target,d);states=[];start=time.perf_counter()
    try:
     for cycle in range(1,P['cycles']+1):
      t=time.perf_counter();modelseed=seed+100000*(cycle-1);m,qd,q,b=fit(cfg,modelseed,x[ids],ys);trainsecs=time.perf_counter()-t;pred=np.einsum('bi,ij,bj->b',x,q,x)+b
      with torch.no_grad():direct=m(torch.tensor(x,dtype=torch.float32)).numpy()
      parity=float(np.max(np.abs(pred-direct)));assert parity<1e-4 and np.isfinite(pred).all()
      h,j=qubo_to_ising_coefficients(q);scale=max([abs(v) for v in h.values()]+[abs(v) for v in j.values()]+[0.]);alpha=tracker.get_current_alpha() if method=='Adaptive-FMQA' else 100.;lam=scale*alpha
      bqm=build_penalty_bqm(qd,[16,3,4],lam,b);sampleseed=seed*100000+cycle*1000;t=time.perf_counter();ss=neal.SimulatedAnnealingSampler().sample(bqm,num_reads=10,num_sweeps=1000,beta_schedule_type='geometric',seed=sampleseed);sasecs=time.perf_counter()-t
      arr=ss.record.sample[:,[list(ss.variables).index(i) for i in range(23)]];bits=np.repeat(arr,ss.record.num_occurrences,axis=0).astype(np.int8);assert len(bits)==10
      chosen,meta=select(bits,pred,x,seen);nextalpha=tracker.update(meta['raw_feasible_rate']) if method=='Adaptive-FMQA' else alpha
      e={'cycle':cycle,'train_ids_before':list(ids),'evaluations_before':calls,'model_seed':modelseed,'model_sha256':model_hash(m),'fm_predictions':pred.tolist(),'offset':b,'base_Ising_scale_S':scale,'alpha':alpha,'lambda':lam,'next_alpha':nextalpha,'sampler_seed':sampleseed,'raw_bits':bits.tolist(),'neal_info':{k:v for k,v in ss.info.items() if k!='timing'},'selected_id':chosen,'selected_prediction':float(pred[chosen]) if chosen is not None else None,'selected_value':None,'train_seconds':trainsecs,'sample_seconds':sasecs,'parity_max_error':parity,**meta}
      states.append({'cycle':cycle,'state_dict':m.state_dict(),'model_sha256':e['model_sha256']})
      if chosen is not None:
       assert chosen not in seen and calls<P['bb_budget_cap'];val=float(bb.evaluate(cats[chosen]));calls+=1;ids.append(chosen);ys.append(val);seen.add(chosen);e['selected_value']=val
      assert calls==len(ids)==len(seen) and calls<=80
      e.update({'evaluations_after':calls,'best_so_far':float(min(ys))});d['events'].append(e);save(target,d)
     cp=OUT/f'data/checkpoints/{name}.pt';torch.save(states,cp);d.update({'status':'completed','final_ids':ids,'final_values':ys,'actual_evaluations':calls,'final_best':float(min(ys)),'checkpoint':str(cp.relative_to(OUT)),'checkpoint_sha256':sha(cp),'total_seconds':time.perf_counter()-start});save(target,d)
     print(name,'completed evaluations',calls,'best',min(ys),flush=True)
    except Exception:
     d.update({'status':'failed','failure':traceback.format_exc(),'actual_evaluations':calls});save(target,d);raise

if __name__=='__main__':globals()[sys.argv[1]]()
