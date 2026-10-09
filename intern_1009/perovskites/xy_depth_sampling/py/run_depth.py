"""Single frozen FM, canonical multi-layer OpenQARP XY depth ablation."""
import sys,json,time,traceback,itertools
from pathlib import Path
import numpy as np
import torch
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'fmqa_adamw_bbo/py'))
from run_bbo import PerovskitesEvaluator,select,sha,save,model_hash,fm_to_qubo,matrix,qubo_to_ising_coefficients,REPO
from fm import TorchFM
from qarp_backend import OpenQARPCompactXYQAOA,OpenQARPXYQAOA
sys.path.insert(0,str(ROOT/'py'))
from validate_xy_ring import independent_basis,independent_edges,reference_evolution
P=json.loads((OUT/'json/protocol.json').read_text())
CODE=[Path(__file__),REPO/'src/fm.py',REPO/'src/fm_to_qubo.py',REPO/'src/qarp_backend.py',ROOT/'fmqa_adamw_bbo/py/run_bbo.py',ROOT/'py/validate_xy_ring.py',ROOT/'py/perovskites_evaluator.py']

def reference_layers(q,sizes,gammas,betas):
 """Independent validation only: ordered edge action, not production solver."""
 labels,x,lookup=independent_basis(sizes);energy=np.einsum('bi,ij,bj->b',x,q,x);state=np.ones(len(x),dtype=complex)/np.sqrt(len(x));edges=independent_edges(sizes)
 for gamma,beta in zip(gammas,betas):
  state*=np.exp(-1j*gamma*energy)
  for group,a,b,_,_ in edges:
   ix=[i for i,label in enumerate(labels) if label[group]==a];jx=[]
   for i in ix:
    other=list(labels[i]);other[group]=b;jx.append(lookup[tuple(other)])
   left=state[ix].copy();right=state[jx].copy();state[ix]=np.cos(beta)*left-1j*np.sin(beta)*right;state[jx]=np.cos(beta)*right-1j*np.sin(beta)*left
 return state

def frozen():
 torch.set_num_threads(1);cp=OUT/'data/fixed_model.pt';assert sha(cp)==P['model_checkpoint_sha256'];m=TorchFM(23,1);m.load_state_dict(torch.load(cp,weights_only=True));m.eval();qd,b=fm_to_qubo(m);q=matrix(qd,23);bb=PerovskitesEvaluator();cats=list(bb.candidates());x=np.array([bb.encode(c) for c in cats],dtype=np.int8);pred=np.einsum('bi,ij,bj->b',x,q,x)+b
 prior=json.loads((ROOT/'single_fm_sampling_quality/json/summary.json').read_text());assert model_hash(m)==prior['model_sha256'];np.testing.assert_array_equal(pred,prior['FM_predictions']);h,j=qubo_to_ising_coefficients(q);s=max([abs(v) for v in h.values()]+[abs(v) for v in j.values()]+[0.]);return m,q,b,x,pred,s,prior['initial_ids']

def prevalidate():
 assert P['status']=='ready';old=json.loads((ROOT/'single_fm_sampling_quality/json/prevalidation.json').read_text());assert old['status']=='passed'
 for n,h in old['source_sha256'].items():assert sha(REPO/n)==h
 checks=[]
 for sizes in [[3,3],[3,3,3],[16,3,4]]:
  n=sum(sizes);q=np.triu(np.random.default_rng(n).normal(0,.1,(n,n)));compact=OpenQARPCompactXYQAOA(q,sizes);full=OpenQARPXYQAOA(q,sizes) if n<10 else None
  for p in P['depths']:
   gammas=np.random.default_rng(n+p).uniform(0,.8,p);betas=np.random.default_rng(100+n+p).uniform(0,.8,p);ref=reference_layers(q,sizes,gammas,betas);prob=compact.feasible_probabilities(gammas,betas);err=float(max(abs(prob-abs(ref)**2)));assert err<1e-10
   fullerr=None;feasible=float(prob.sum())
   if full:
    state=full.statevector(gammas,betas);fs=state[full.feasible_indices];fullerr=float(max(abs(abs(fs)**2-prob)));assert fullerr<1e-10 and abs(1-np.vdot(state,state).real)<1e-10 and abs(1-np.vdot(fs,fs).real)<1e-10
   assert abs(1-feasible)<1e-10;checks.append({'groups':sizes,'p':p,'reference_probability_error':err,'full_compact_probability_error':fullerr,'raw_feasible_mass':feasible})
 m,q,b,x,pred,s,ids=frozen();div=s if s>0 else 1.;backend=OpenQARPCompactXYQAOA(q/div,[16,3,4]);prev=json.loads((ROOT/'initial_data_sensitivity/json/XY-FMQAOA_initial10_seed42.json').read_text())['events'][0];oldgrid=[]
 for gamma,beta in itertools.product([.05,.4,.8],[.1,.4,.8]):
  prob=backend.feasible_probabilities([gamma],[beta]);ref,_,_=reference_evolution(q/div,[16,3,4],gamma,beta);np.testing.assert_allclose(prob,abs(ref)**2,atol=1e-10,rtol=0);value=float(prob@(pred/div));oldgrid.append({'gammas':[gamma],'betas':[beta],'expected_FM':value*div})
 best=min(oldgrid,key=lambda z:z['expected_FM']);assert best['gammas']+best['betas']==prev['chosen_angles'];np.testing.assert_allclose(backend.feasible_probabilities(best['gammas'],best['betas']),prev['feasible_probabilities'],atol=1e-10,rtol=0)
 save(OUT/'json/prevalidation.json',{'status':'passed','cases':checks,'old9_p1_reproduction':'passed','old9_p1_best':best,'fixed_model_hash':model_hash(m),'source_sha256':{str(c.relative_to(REPO)):sha(c) for c in CODE}});print('p1/2/3 N6/N9/N23 validation passed',flush=True)

def angles(p,seed,previous):
 rng=np.random.default_rng(np.random.SeedSequence([42,1009,p,seed,91]));result=[]
 if p==1:result=[([g],[b]) for g,b in itertools.product([.05,.4,.8],[.1,.4,.8])]
 else:result=[(previous['gammas']+[0.],previous['betas']+[0.])]
 while len(result)<128:result.append((rng.uniform(0,.8,p).tolist(),rng.uniform(0,.8,p).tolist()))
 return result

def run():
 v=json.loads((OUT/'json/prevalidation.json').read_text());assert v['status']=='passed' and v['source_sha256']=={str(c.relative_to(REPO)):sha(c) for c in CODE};m,q,b,x,pred,s,ids=frozen();div=s if s>0 else 1.;backend=OpenQARPCompactXYQAOA(q/div,[16,3,4]);order=np.argsort(x.astype(np.uint64)@(1<<np.arange(23,dtype=np.uint64)));umin=float(min(pred[i] for i in range(192) if i not in ids));minimum=[i for i in range(192) if i not in ids and abs(pred[i]-umin)<1e-8]
 env=json.loads((ROOT/'large_penalty_diagnostic/json/environment.json').read_text());env.update(backend='canonical OpenQARPCompactXYQAOA',source_sha256=v['source_sha256'],protocol_sha256=sha(OUT/'json/protocol.json'));save(OUT/'json/environment.json',env)
 for seed in range(10):
  previous=None
  for p in P['depths']:
   path=OUT/f'json/searchseed{seed}_p{p}.json';assert not path.exists();d={'status':'running','search_seed':seed,'p':p,'lambda':0,'S':s,'normalization_divisor':div,'model_sha256':model_hash(m),'protocol_sha256':sha(OUT/'json/protocol.json'),'search':[],'measurements':[]};save(path,d);start=time.perf_counter()
   try:
    best=None
    for k,(gs,bs) in enumerate(angles(p,seed,previous)):
     t=time.perf_counter();prob=backend.feasible_probabilities(gs,bs);value=float(prob@pred);row={'index':k,'gammas':gs,'betas':bs,'expected_FM':value,'feasible_mass':float(prob.sum()),'seconds':time.perf_counter()-t};d['search'].append(row)
     if best is None or value<best['expected_FM']:best=row.copy();bestprob=prob.copy()
     if (k+1)%16==0:save(path,d)
    if previous:assert best['expected_FM']<=previous['expected_FM']+1e-10
    t=time.perf_counter();verified=backend.feasible_probabilities(best['gammas'],best['betas']);np.testing.assert_allclose(verified,bestprob,atol=1e-12,rtol=0);d.update(best_angles=best,probabilities=bestprob.tolist(),selected_state_verification_seconds=time.perf_counter()-t);cdf=np.cumsum(bestprob[order]);minimum_mass=float(bestprob[minimum].sum())
    for rep in range(10):
     rng=np.random.default_rng(np.random.SeedSequence([42,1009,seed,rep,93]));u=rng.random(100)*cdf[-1];draws=order[np.searchsorted(cdf,u,side='left')];chosen,filters=select(x[draws],pred,x,set(ids));d['measurements'].append({'rep':rep,'raw_candidate_ids':draws.tolist(),'selected_id':chosen,'selected_unseen_gap':float(pred[chosen]-umin) if chosen is not None else None,'minimum_hit':bool(any(i in minimum for i in draws)),**filters})
    d.update(status='completed',ideal_unseen_mass=float(sum(bestprob[i] for i in range(192) if i not in ids)),ideal_minimum_mass=minimum_mass,ideal_100shot_minimum_capture=1-(1-minimum_mass)**100,expected_FM_gap_to_global_min=best['expected_FM']-float(pred.min()),total_seconds=time.perf_counter()-start);save(path,d);previous=best;print('search',seed,'p',p,'expected_gap',d['expected_FM_gap_to_global_min'],'100shot_capture',d['ideal_100shot_minimum_capture'],flush=True)
   except Exception:d.update(status='failed',traceback=traceback.format_exc());save(path,d);raise

if __name__=='__main__':globals()[sys.argv[1]]()
