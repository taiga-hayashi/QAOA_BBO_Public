"""User-requested initial3 closed-loop p3 and alpha1000 comparison."""
import sys,json,time,traceback,itertools
from pathlib import Path
import numpy as np
import torch
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'large_penalty_diagnostic/py'))
from run_penalty import fit,CFG,event,sample,problem,sha,save,model_hash,REPO,qubo_to_ising_coefficients,build_penalty_bqm
sys.path.insert(0,str(ROOT/'xy_depth_sampling/py'))
from run_depth import reference_layers,OpenQARPCompactXYQAOA
P=json.loads((OUT/'json/protocol.json').read_text())
CODE=[Path(__file__),ROOT/'large_penalty_diagnostic/py/run_penalty.py',ROOT/'xy_depth_sampling/py/run_depth.py',ROOT/'fmqa_adamw_bbo/py/run_bbo.py',REPO/'src/fm.py',REPO/'src/fm_to_qubo.py',REPO/'src/qarp_backend.py',REPO/'intern_0924/src/penalty.py',ROOT/'py/perovskites_evaluator.py',ROOT/'py/validate_xy_ring.py']

def initial(seed):return json.loads((ROOT/f'fixed_fm_pilot/json/seed_{seed}.json').read_text())['initial_dataset']['candidate_ids'][:3]

def candidates(p,seed,cycle):
 result=[([g]+[0.]*(p-1),[b]+[0.]*(p-1)) for g,b in itertools.product([.05,.4,.8],[.1,.4,.8])];rng=np.random.default_rng(np.random.SeedSequence([seed,1009,p,cycle,95]))
 while len(result)<128:result.append((rng.uniform(0,.8,p).tolist(),rng.uniform(0,.8,p).tolist()))
 return result

def generate(qd,q,b,pred,method,seed,cycle,x):
 rngseed=seed*100000+cycle*1000
 if method=='LargePenalty-alpha1000':return sample(qd,q,b,1000,rngseed)
 p=1 if method=='XY-p1' else 3;h,j=qubo_to_ising_coefficients(q);s=max([abs(v) for v in h.values()]+[abs(v) for v in j.values()]+[0.]);div=s if s>0 else 1.
 # Unique continuous angles do not benefit from retaining caches across cycles.
 OpenQARPCompactXYQAOA._mixer_cache.clear();backend=OpenQARPCompactXYQAOA(q/div,[16,3,4]);search=[];best=None
 for index,(gs,bs) in enumerate(candidates(p,seed,cycle)):
  prob=backend.feasible_probabilities(gs,bs);value=float(prob@pred);row={'index':index,'gammas':gs,'betas':bs,'expected_FM':value,'feasible_mass':float(prob.sum())};search.append(row)
  if best is None or value<best['expected_FM']:best=row;bestprob=prob.copy()
 order=np.argsort(x.astype(np.uint64)@(1<<np.arange(23,dtype=np.uint64)));cdf=np.cumsum(bestprob[order]);u=np.random.default_rng(rngseed).random(100)*cdf[-1];draws=order[np.searchsorted(cdf,u,side='left')]
 return x[draws],{'p':p,'alpha':0.,'lambda':0.,'base_Ising_scale_S':s,'normalization_divisor':div,'sampler_seed':rngseed,'angle_search':search,'best_angles':best,'feasible_probabilities':bestprob.tolist(),'raw_candidate_ids':draws.tolist()}

def prevalidate():
 assert P['status']=='ready';old=json.loads((ROOT/'xy_depth_sampling/json/prevalidation.json').read_text());assert old['status']=='passed'
 for n,h in old['source_sha256'].items():assert sha(REPO/n)==h
 assert len([c for c in old['cases'] if c['groups'] in [[3,3],[3,3,3]] and c['p']==3])==2
 bb,cats,x=problem();assert bb.data_sha256==P['data_sha256'];checks=[]
 for seed in P['seeds']:
  ids=initial(seed);assert len(ids)==len(set(ids))==3;vals=[float(bb.evaluate(cats[i])) for i in ids];m,qd,q,b=fit(CFG,seed,x[ids],vals);pred=np.einsum('bi,ij,bj->b',x,q,x)+b
  a=OpenQARPCompactXYQAOA(q,[16,3,4]);b3=OpenQARPCompactXYQAOA(q,[16,3,4])
  for g,bt in itertools.product([.05,.4,.8],[.1,.4,.8]):np.testing.assert_allclose(a.feasible_probabilities([g],[bt]),b3.feasible_probabilities([g,0.,0.],[bt,0.,0.]),atol=1e-10,rtol=0)
  for method in P['methods']:
   bits,meta=generate(qd,q,b,pred,method,seed,1,x);e=event(m,q,b,bits,meta,x,ids);assert len(bits)==100
   if method.startswith('XY'):
    best=e['best_angles'];prob=abs(reference_layers(q/e['normalization_divisor'],[16,3,4],best['gammas'],best['betas']))**2;np.testing.assert_allclose(prob,e['feasible_probabilities'],atol=1e-10,rtol=0);assert e['raw_feasible_rate']==1 and e['lambda']==0
   else:
    xx=np.concatenate([x,np.random.default_rng(seed).integers(0,2,(50,23))]);v=sum((xx[:,a:z].sum(1)-1)**2 for a,z in [(0,16),(16,19),(19,23)]);ener=np.einsum('bi,ij,bj->b',xx,q,xx)+b;np.testing.assert_allclose(build_penalty_bqm(qd,[16,3,4],e['lambda'],b).energies((xx,list(range(23)))),ener+e['lambda']*v,atol=1e-8,rtol=1e-11)
  checks.append({'seed':seed,'initial_ids':ids,'initial_values':vals,'model_hash':model_hash(m),'three_methods_initial_FM_and_p3_anchor_checks':'passed'})
 save(OUT/'json/prevalidation.json',{'status':'passed','inherited_N6_N9_N23_p1_p2_p3':'passed; unchanged source hashes','initial3_checks':checks,'source_sha256':{str(c.relative_to(REPO)):sha(c) for c in CODE}});print('initial3 p1/p3/alpha1000 prevalidation passed',flush=True)

def run():
 v=json.loads((OUT/'json/prevalidation.json').read_text());assert v['status']=='passed' and v['source_sha256']=={str(c.relative_to(REPO)):sha(c) for c in CODE};bb,cats,x=problem();env=json.loads((ROOT/'xy_depth_sampling/json/environment.json').read_text());env.update(source_sha256=v['source_sha256'],protocol_sha256=sha(OUT/'json/protocol.json'),backend='XY canonical OpenQARP exact encoded,SA Neal');save(OUT/'json/environment.json',env)
 for seed in P['seeds']:
  for method in P['methods']:
   name=f'{method}_seed{seed}';path=OUT/f'json/{name}.json';cp=OUT/f'data/checkpoints/{name}.pt';assert not path.exists();ids=initial(seed);vals=[float(bb.evaluate(cats[i])) for i in ids];d={'status':'running','method':method,'seed':seed,'initial_ids':list(ids),'initial_values':list(vals),'protocol_sha256':sha(OUT/'json/protocol.json'),'events':[]};states=[];start=time.perf_counter();save(path,d)
   try:
    for cycle in range(1,78):
     t=time.perf_counter();ms=seed+100000*(cycle-1);m,qd,q,b=fit(CFG,ms,x[ids],vals);training=time.perf_counter()-t;pred=np.einsum('bi,ij,bj->b',x,q,x)+b;t=time.perf_counter();bits,meta=generate(qd,q,b,pred,method,seed,cycle,x);e=event(m,q,b,bits,meta,x,ids);e.update(cycle=cycle,model_seed=ms,evaluations_before=len(ids),training_seconds=training,generation_seconds=time.perf_counter()-t)
     states.append({'cycle':cycle,'state_dict':m.state_dict(),'model_hash':model_hash(m)})
     if e['selected_id'] is not None:
      i=e['selected_id'];assert i not in ids and len(ids)<80;value=float(bb.evaluate(cats[i]));ids.append(i);vals.append(value);e['selected_value']=value
     assert len(ids)==len(set(ids))<=80;e.update(evaluations_after=len(ids),best_so_far=min(vals));d['events'].append(e);save(path,d)
     if cycle%10==0:torch.save(states,cp)
    torch.save(states,cp);d.update(status='completed',final_ids=ids,final_values=vals,actual_evaluations=len(ids),final_best=min(vals),checkpoint=str(cp.relative_to(OUT)),checkpoint_sha256=sha(cp),total_seconds=time.perf_counter()-start);save(path,d);print(name,'completed best',min(vals),'evals',len(ids),flush=True)
   except Exception:
    torch.save(states,cp);d.update(status='failed',traceback=traceback.format_exc(),checkpoint=str(cp.relative_to(OUT)),checkpoint_sha256=sha(cp));save(path,d);raise

if __name__=='__main__':globals()[sys.argv[1]]()
