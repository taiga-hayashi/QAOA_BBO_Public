"""Canonical OpenQARP exact compact XY closed-loop lookup pilot."""
import sys,json,time,traceback
from pathlib import Path
import numpy as np
import torch
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'fmqa_adamw_bbo/py'))
from run_bbo import fit,select,PerovskitesEvaluator,save,sha,model_hash,REPO,qubo_to_ising_coefficients
from qarp_backend import OpenQARPCompactXYQAOA,OpenQARPXYQAOA
sys.path.insert(0,str(ROOT/'py'))
from validate_xy_ring import reference_evolution
P=json.loads((OUT/'json/protocol.json').read_text());CFG={'id':'proposed','rank':1,'optimizer':'AdamW','weight_decay':.01}
CODE=[Path(__file__),REPO/'src/qarp_backend.py',REPO/'src/fm.py',REPO/'src/fm_to_qubo.py',ROOT/'fmqa_adamw_bbo/py/run_bbo.py',ROOT/'py/perovskites_evaluator.py']

def prevalidate():
 assert (REPO/'src/qarp_backend.py').read_bytes()==(REPO/'intern/src/qarp_backend.py').read_bytes()
 bb=PerovskitesEvaluator();cats=list(bb.candidates());x=np.array([bb.encode(c) for c in cats]);records=[]
 for sizes in [[3,3],[3,3,3],[16,3,4]]:
  n=sum(sizes)
  if n==23:
   base=json.loads((ROOT/'fixed_fm_pilot/json/seed_42.json').read_text());ids=base['initial_dataset']['candidate_ids'];m,qd,q,b=fit(CFG,42,x[ids],[bb.evaluate(cats[i]) for i in ids]);h,j=qubo_to_ising_coefficients(q);scale=max([abs(v) for v in h.values()]+[abs(v) for v in j.values()]);q=q/scale
  else:q=np.triu(np.random.default_rng(n).normal(size=(n,n)))
  full=OpenQARPXYQAOA(q,sizes);compact=OpenQARPCompactXYQAOA(q,sizes)
  angles=[(g,b) for g in P['qaoa']['gammas'] for b in P['qaoa']['betas']]
  for gamma,beta in angles:
   f=full.statevector([gamma],[beta]);c=compact.statevector([gamma],[beta]);fp=np.abs(f[full.feasible_indices])**2;cp=np.abs(c[compact.encoded_indices])**2
   ref,_,_=reference_evolution(q,sizes,gamma,beta);ce=c[compact.encoded_indices];phase=np.vdot(ref,ce);phase/=abs(phase)
   err=float(np.max(np.abs(fp-cp)));amp=float(np.max(np.abs(ce-ref*phase)));mass=float(cp.sum());norm=float(np.vdot(c,c).real);fmass=float(fp.sum());assert err<1e-10 and amp<1e-10 and abs(1-mass)<1e-10 and abs(1-norm)<1e-10 and abs(1-fmass)<1e-10
   records.append({'N':n,'gamma':gamma,'beta':beta,'probability_max_error':err,'reference_amplitude_max_error_up_to_global_phase':amp,'compact_norm':norm,'compact_feasible_mass':mass,'full_feasible_mass':fmass});print('validated',n,gamma,beta,err,flush=True)
  if n==9:
   f=full.statevector([.4,-.2],[.8,.3]);c=compact.statevector([.4,-.2],[.8,.3]);np.testing.assert_allclose(np.abs(f[full.feasible_indices])**2,np.abs(c[compact.encoded_indices])**2,atol=1e-10,rtol=0)
 save(OUT/'json/prevalidation.json',{'status':'passed','cases':records,'p2_N9':'passed','source_sha256':{str(c.relative_to(REPO)):sha(c) for c in CODE},'shared_source_mirror':'equal'})

def run():
 v=json.loads((OUT/'json/prevalidation.json').read_text());assert v['source_sha256']=={str(c.relative_to(REPO)):sha(c) for c in CODE}
 bb=PerovskitesEvaluator();assert bb.data_sha256=='ba55731336cf68a2fcc2a8c6b542ee06d4c238c91ced27d243ce9de95f8d6518';cats=[list(c) for c in bb.candidates()];x=np.array([bb.encode(c) for c in cats],dtype=np.int8);basis=x.astype(np.uint64)@(1<<np.arange(23,dtype=np.uint64));order=np.argsort(basis)
 env=json.loads((ROOT/'fmqa_adamw_bbo/json/environment.json').read_text());env.update({'protocol_sha256':sha(OUT/'json/protocol.json'),'code_sha256':v['source_sha256'],'backend':'OpenQARPCompactXYQAOA exact encoded simulation','original_qubits':23,'simulator_qubits':8});save(OUT/'json/environment.json',env)
 for seed in P['seeds']:
  target=OUT/f'json/seed_{seed}.json';assert not target.exists();basepath=ROOT/f'fixed_fm_pilot/json/seed_{seed}.json';base=json.loads(basepath.read_text());ids=list(base['initial_dataset']['candidate_ids']);ys=[float(bb.evaluate(cats[i])) for i in ids];states=[];d={'seed':seed,'status':'running','initial_ids':list(ids),'initial_values':list(ys),'initial_sha256':base['initial_sha256'],'source_initial_sha256':sha(basepath),'events':[]};start=time.perf_counter();save(target,d)
  try:
   for cycle in range(1,61):
    t=time.perf_counter();m,qd,q,b=fit(CFG,seed+100000*(cycle-1),x[ids],ys);pred=np.einsum('bi,ij,bj->b',x,q,x)+b;trainsecs=time.perf_counter()-t
    with torch.no_grad():parity=float(np.max(np.abs(pred-m(torch.tensor(x,dtype=torch.float32)).numpy())))
    assert parity<1e-4;h,j=qubo_to_ising_coefficients(q);scale=max([abs(v) for v in h.values()]+[abs(v) for v in j.values()]+[0.]);divisor=scale if scale>0 else 1.;backend=OpenQARPCompactXYQAOA(q/divisor,[16,3,4]);search=[];best=None;t=time.perf_counter()
    for gamma in P['qaoa']['gammas']:
     for beta in P['qaoa']['betas']:
      prob=backend.feasible_probabilities([gamma],[beta]);value=float(prob@(pred/divisor));search.append({'gamma':gamma,'beta':beta,'expected_normalized_FM':value,'feasible_mass':float(prob.sum())})
      if best is None or value<best[0]:best=(value,gamma,beta,prob.copy())
    prob=best[3];shotseed=seed*100000+cycle*1000;cdf=np.cumsum(prob[order]);u=np.random.default_rng(shotseed).random(10)*cdf[-1];sampleids=order[np.searchsorted(cdf,u,side='left')];bits=x[sampleids];chosen,meta=select(bits,pred,x,set(ids));gen=time.perf_counter()-t
    e={'cycle':cycle,'train_ids_before':list(ids),'evaluations_before':len(ids),'model_seed':seed+100000*(cycle-1),'model_sha256':model_hash(m),'fm_predictions':pred.tolist(),'base_Ising_scale_S':scale,'normalization_divisor':divisor,'lambda':0.,'alpha':0.,'angle_search':search,'chosen_angles':[best[1],best[2]],'feasible_probabilities':prob.tolist(),'shot_seed':shotseed,'raw_candidate_ids':sampleids.tolist(),'raw_bits':bits.tolist(),'selected_id':chosen,'selected_prediction':float(pred[chosen]) if chosen is not None else None,'selected_value':None,'train_seconds':trainsecs,'generation_seconds':gen,'parity_max_error':parity,**meta};states.append({'cycle':cycle,'state_dict':m.state_dict(),'model_sha256':model_hash(m)})
    if chosen is not None:
     assert chosen not in ids and len(ids)<80;value=float(bb.evaluate(cats[chosen]));ids.append(chosen);ys.append(value);e['selected_value']=value
    e.update({'evaluations_after':len(ids),'best_so_far':min(ys)});d['events'].append(e);save(target,d)
   cp=OUT/f'data/checkpoints/seed_{seed}.pt';torch.save(states,cp);d.update({'status':'completed','final_ids':ids,'final_values':ys,'final_best':min(ys),'actual_evaluations':len(ids),'checkpoint':str(cp.relative_to(OUT)),'checkpoint_sha256':sha(cp),'total_seconds':time.perf_counter()-start});save(target,d);print('completed',seed,'best',min(ys),'evals',len(ids),flush=True)
  except Exception:d.update({'status':'failed','error':traceback.format_exc()});save(target,d);raise

if __name__=='__main__':globals()[sys.argv[1]]()
