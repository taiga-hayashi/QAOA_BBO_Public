"""Large-alpha diagnostic using canonical FM, QUBO, Neal and selection."""
import sys,json,time,traceback
from pathlib import Path
import numpy as np
import torch
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'fmqa_adamw_bbo/py'))
from run_bbo import fit,select,PerovskitesEvaluator,save,sha,model_hash,matrix,build_penalty_bqm,qubo_to_ising_coefficients,REPO,neal
P=json.loads((OUT/'json/protocol.json').read_text());CFG={'rank':1,'optimizer':'AdamW','weight_decay':.01}
CODE=[Path(__file__),ROOT/'fmqa_adamw_bbo/py/run_bbo.py',REPO/'src/fm.py',REPO/'src/fm_to_qubo.py',REPO/'src/fmqa_solver.py',REPO/'src/qarp_backend.py',REPO/'intern_0924/src/penalty.py',ROOT/'py/perovskites_evaluator.py']

def problem():
 bb=PerovskitesEvaluator();assert bb.data_sha256==P['data_sha256'];cats=list(bb.candidates());x=np.array([bb.encode(c) for c in cats],dtype=np.int8);return bb,cats,x

def initial(seed):return json.loads((ROOT/f'fixed_fm_pilot/json/seed_{seed}.json').read_text())['initial_dataset']['candidate_ids'][:10]

def sample(qd,q,b,alpha,seed,beta=None):
 h,j=qubo_to_ising_coefficients(q);s=max([abs(v) for v in h.values()]+[abs(v) for v in j.values()]+[0.]);lam=s*alpha
 ss=neal.SimulatedAnnealingSampler().sample(build_penalty_bqm(qd,[16,3,4],lam,b),num_reads=100,num_sweeps=1000,beta_schedule_type='geometric',seed=seed,**({'beta_range':beta} if beta is not None else {}))
 arr=ss.record.sample[:,[list(ss.variables).index(i) for i in range(23)]];bits=np.repeat(arr,ss.record.num_occurrences,axis=0).astype(np.int8)
 assert bits.shape==(100,23)
 return bits,{'alpha':alpha,'lambda':lam,'base_Ising_scale_S':s,'sampler_seed':seed,'neal_info':{k:v for k,v in ss.info.items() if k!='timing'}}

def event(m,q,b,bits,meta,x,ids):
 pred=np.einsum('bi,ij,bj->b',x,q,x)+b
 with torch.no_grad():err=float(np.max(np.abs(pred-m(torch.tensor(x,dtype=torch.float32)).numpy())))
 assert err<1e-4 and np.isfinite(pred).all();chosen,filters=select(bits,pred,x,set(ids));umin=float(min(pred[i] for i in range(192) if i not in ids))
 return {'model_sha256':model_hash(m),'train_ids_before':list(ids),'fm_predictions':pred.tolist(),'raw_bits':bits.tolist(),'selected_id':chosen,'selected_prediction':float(pred[chosen]) if chosen is not None else None,'selected_unseen_fm_gap':float(pred[chosen]-umin) if chosen is not None else None,'selected_value':None,'unseen_fm_min':umin,'parity_max_error':err,**meta,**filters}

def prevalidate():
 assert P['status']=='ready';old=json.loads((ROOT/'initial_data_sensitivity/json/prevalidation.json').read_text());assert old['status']=='passed'
 for n,h in old['source_sha256'].items():assert sha(REPO/n)==h
 # Exhaustive small-bit penalty checks, including extreme alpha.
 for groups in [[3,3],[3,3,3]]:
  n=sum(groups);xx=((np.arange(1<<n)[:,None]>>np.arange(n))&1).astype(np.int8);q=np.triu(np.random.default_rng(n).normal(size=(n,n)));qd={(i,j):float(q[i,j]) for i in range(n) for j in range(i,n)}
  starts=np.cumsum([0]+groups);v=sum((xx[:,a:z].sum(1)-1)**2 for a,z in zip(starts[:-1],starts[1:]));base=np.einsum('bi,ij,bj->b',xx,q,xx)+.3
  for lam in [0.,1.,1e6]:np.testing.assert_allclose(build_penalty_bqm(qd,groups,lam,.3).energies((xx,list(range(n)))),base+lam*v,atol=1e-7,rtol=1e-12)
 bb,cats,x=problem();checks=[]
 for seed in P['seeds']:
  ids=initial(seed);m,qd,q,b=fit(CFG,seed,x[ids],[bb.evaluate(cats[i]) for i in ids]);baseline=json.loads((ROOT/f'initial_data_sensitivity/json/LargePenalty-FMQA_initial10_seed{seed}.json').read_text())['events'][0]
  bits,meta=sample(qd,q,b,100,seed*100000+1000);e=event(m,q,b,bits,meta,x,ids);assert bits.tolist()==baseline['raw_bits'] and e['selected_id']==baseline['selected_id'] and model_hash(m)==baseline['model_sha256'];np.testing.assert_array_equal(e['fm_predictions'],baseline['fm_predictions'])
  xx=np.concatenate([x,np.random.default_rng(seed).integers(0,2,(100,23))]);v=sum((xx[:,a:z].sum(1)-1)**2 for a,z in [(0,16),(16,19),(19,23)]);base=np.einsum('bi,ij,bj->b',xx,q,xx)+b
  maxerr=0.
  for alpha in P['alphas']:
   lam=meta['base_Ising_scale_S']*alpha;got=build_penalty_bqm(qd,[16,3,4],lam,b).energies((xx,list(range(23))));maxerr=max(maxerr,float(np.max(abs(got[:192]-base[:192]))));np.testing.assert_allclose(got,base+lam*v,atol=1e-6,rtol=1e-11)
  checks.append({'seed':seed,'firstcycle100':'exact','extreme_feasible_energy_roundoff_eV':maxerr})
 save(OUT/'json/prevalidation.json',{'status':'passed','inherited_N6_N9_N23':'passed; unchanged source hashes','exhaustive_N6_N9_penalty':'passed','checks':checks,'source_sha256':{str(c.relative_to(REPO)):sha(c) for c in CODE}});print('prevalidation passed',flush=True)

def audit():
 v=json.loads((OUT/'json/prevalidation.json').read_text());assert v['status']=='passed' and v['source_sha256']=={str(c.relative_to(REPO)):sha(c) for c in CODE}
 env=json.loads((ROOT/'initial_data_sensitivity/json/environment.json').read_text());env.update(source_sha256=v['source_sha256'],protocol_sha256=sha(OUT/'json/protocol.json'),backend='classical Neal SA; no QAOA execution');save(OUT/'json/environment.json',env)

def fixed():
 audit();bb,cats,x=problem()
 for seed in P['seeds']:
  path=OUT/f'json/fixed_seed{seed}.json';assert not path.exists();ids=initial(seed);vals=[float(bb.evaluate(cats[i])) for i in ids];m,qd,q,b=fit(CFG,seed,x[ids],vals);cp=OUT/f'data/checkpoints/fixed_seed{seed}.pt';torch.save(m.state_dict(),cp)
  _,ref=sample(qd,q,b,1,seed*100000+17);beta=ref['neal_info']['beta_range'];d={'seed':seed,'status':'running','initial_ids':ids,'initial_values':vals,'protocol_sha256':sha(OUT/'json/protocol.json'),'checkpoint':str(cp.relative_to(OUT)),'checkpoint_sha256':sha(cp),'reference_beta_range':beta,'events':[]};save(path,d)
  try:
   for rep in range(10):
    for schedule in ['auto','fixed_alpha1_beta']:
     for alpha in P['alphas']:
      t=time.perf_counter();bits,meta=sample(qd,q,b,alpha,seed*100000+rep*1000+17,beta if schedule!='auto' else None);e=event(m,q,b,bits,meta,x,ids);e.update(rep=rep,schedule=schedule,generation_seconds=time.perf_counter()-t)
      if e['selected_id'] is not None:e['selected_value']=float(bb.evaluate(cats[e['selected_id']]))
      d['events'].append(e);save(path,d)
   d['status']='completed';save(path,d);print('fixed',seed,'140batches complete',flush=True)
  except Exception:d.update(status='failed',traceback=traceback.format_exc());save(path,d);raise

def bbo():
 audit();bb,cats,x=problem()
 for seed in P['seeds']:
  for alpha in P['alphas']:
   name=f'bbo_alpha{alpha}_seed{seed}';path=OUT/f'json/{name}.json';assert not path.exists();ids=initial(seed);vals=[float(bb.evaluate(cats[i])) for i in ids];d={'status':'running','seed':seed,'alpha':alpha,'initial_ids':list(ids),'initial_values':list(vals),'protocol_sha256':sha(OUT/'json/protocol.json'),'events':[]};states=[];save(path,d);start=time.perf_counter()
   try:
    for cycle in range(1,71):
     t=time.perf_counter();ms=seed+100000*(cycle-1);m,qd,q,b=fit(CFG,ms,x[ids],vals);training=time.perf_counter()-t;t=time.perf_counter();bits,meta=sample(qd,q,b,alpha,seed*100000+cycle*1000);e=event(m,q,b,bits,meta,x,ids);e.update(cycle=cycle,model_seed=ms,evaluations_before=len(ids),training_seconds=training,generation_seconds=time.perf_counter()-t)
     states.append({'cycle':cycle,'state_dict':m.state_dict(),'model_sha256':model_hash(m)})
     if e['selected_id'] is not None:
      i=e['selected_id'];assert i not in ids and len(ids)<80;value=float(bb.evaluate(cats[i]));ids.append(i);vals.append(value);e['selected_value']=value
     e.update(evaluations_after=len(ids),best_so_far=min(vals));assert len(ids)==len(set(ids))<=80;d['events'].append(e);save(path,d)
    cp=OUT/f'data/checkpoints/{name}.pt';torch.save(states,cp);d.update(status='completed',final_ids=ids,final_values=vals,actual_evaluations=len(ids),final_best=min(vals),checkpoint=str(cp.relative_to(OUT)),checkpoint_sha256=sha(cp),total_seconds=time.perf_counter()-start);save(path,d);print(name,'best',min(vals),'evals',len(ids),flush=True)
   except Exception:d.update(status='failed',traceback=traceback.format_exc());save(path,d);raise

if __name__=='__main__':globals()[sys.argv[1]]()
