"""Matched initial3 exact-FM and random-unseen closed-loop controls."""
import sys,json,time,traceback
from pathlib import Path
import numpy as np
import torch
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'xy_p3_initial3_bbo/py'))
from run_initial3 import fit,CFG,initial,problem,sha,save,model_hash,CODE,REPO,OUT as BASE
from run_bbo import select
P=json.loads((OUT/'json/protocol.json').read_text())
FILES=list(dict.fromkeys([Path(__file__),*CODE]))

def pick(method,pred,x,ids,seed,cycle):
 unseen=np.array([i for i in range(len(x)) if i not in ids])
 if method=='Exact-FM':chosen,_=select(x,pred,x,set(ids));return chosen
 return int(np.random.default_rng(seed*100000+cycle*1000).choice(unseen))

def prevalidate():
 v=json.loads((BASE/'json/prevalidation.json').read_text());assert v['status']=='passed'
 assert v['source_sha256']=={str(c.relative_to(REPO)):sha(c) for c in CODE}
 assert sha(BASE/'json/protocol.json')==P['baseline_protocol_sha256'];bb,cats,x=problem();assert bb.data_sha256==P['data_sha256']
 assert pick('Exact-FM',np.zeros(192),x,[0],42,1)==min(range(1,192),key=lambda i:tuple(x[i]))
 checks=[]
 for seed in P['seeds']:
  ids=initial(seed);ys=[float(bb.evaluate(cats[i])) for i in ids];m,qd,q,b=fit(CFG,seed,x[ids],ys);pred=np.einsum('bi,ij,bj->b',x,q,x)+b
  with torch.no_grad():np.testing.assert_allclose(pred,m(torch.tensor(x,dtype=torch.float32)).numpy(),atol=1e-4,rtol=0)
  assert model_hash(m)==next(z for z in v['initial3_checks'] if z['seed']==seed)['model_hash']
  chosen=pick('Exact-FM',pred,x,ids,seed,1);unseen=[i for i in range(192) if i not in ids];assert chosen==min(unseen,key=lambda i:(pred[i],tuple(x[i])))
  r=pick('Random-unseen',pred,x,ids,seed,1);assert r in unseen and r==pick('Random-unseen',pred,x,ids,seed,1)
  checks.append({'seed':seed,'model_sha256':model_hash(m),'exact_first_id':chosen,'random_first_id':r})
 for item in P['baseline_files']:assert sha(ROOT/item['path'])==item['sha256']
 save(OUT/'json/prevalidation.json',{'status':'passed','checks':checks,'source_sha256':{str(c.relative_to(REPO)):sha(c) for c in FILES},'inherited_N6_N9_N23':'passed unchanged source; no new quantum circuit'})
 print('prevalidation passed',flush=True)

def run():
 assert P['status']=='ready';v=json.loads((OUT/'json/prevalidation.json').read_text());assert v['status']=='passed' and v['source_sha256']=={str(c.relative_to(REPO)):sha(c) for c in FILES};bb,cats,x=problem();assert bb.data_sha256==P['data_sha256']
 for seed in P['seeds']:
  for method in P['methods']:
   name=f'{method}_seed{seed}';path=OUT/f'json/{name}.json';cp=OUT/f'data/checkpoints/{name}.pt';assert not path.exists(),'preserve prior/partial runs';ids=initial(seed);ys=[float(bb.evaluate(cats[i])) for i in ids];d={'status':'running','method':method,'seed':seed,'initial_ids':list(ids),'initial_values':list(ys),'events':[],'protocol_sha256':sha(OUT/'json/protocol.json')};states=[];save(path,d)
   try:
    for cycle in range(1,78):
     ms=seed+100000*(cycle-1);t=time.perf_counter();m,qd,q,b=fit(CFG,ms,x[ids],ys);train=time.perf_counter()-t;pred=np.einsum('bi,ij,bj->b',x,q,x)+b
     with torch.no_grad():assert np.max(abs(pred-m(torch.tensor(x,dtype=torch.float32)).numpy()))<1e-4
     t=time.perf_counter();chosen=pick(method,pred,x,ids,seed,cycle);gen=time.perf_counter()-t;assert chosen not in ids and len(ids)<80;e={'cycle':cycle,'model_seed':ms,'model_sha256':model_hash(m),'train_ids_before':list(ids),'evaluations_before':len(ids),'fm_predictions':pred.tolist(),'selected_id':chosen,'sampler_seed':seed*100000+cycle*1000,'training_seconds':train,'generation_seconds':gen};states.append({'cycle':cycle,'state_dict':m.state_dict(),'model_hash':model_hash(m)})
     value=float(bb.evaluate(cats[chosen]));ids.append(chosen);ys.append(value);e.update(selected_value=value,evaluations_after=len(ids),best_so_far=min(ys));d['events'].append(e);save(path,d)
     if cycle%10==0:torch.save(states,cp)
    torch.save(states,cp);d.update(status='completed',final_ids=ids,final_values=ys,actual_evaluations=len(ids),final_best=min(ys),checkpoint=str(cp.relative_to(OUT)),checkpoint_sha256=sha(cp));save(path,d);print(name,'completed',min(ys),flush=True)
   except Exception:
    torch.save(states,cp);d.update(status='failed',traceback=traceback.format_exc(),checkpoint=str(cp.relative_to(OUT)),checkpoint_sha256=sha(cp));save(path,d);raise

if __name__=='__main__':globals()[sys.argv[1]]()
