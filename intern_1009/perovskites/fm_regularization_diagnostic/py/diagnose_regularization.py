"""Fixed-data Adam/AdamW diagnostic using the canonical shared trainer."""
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr,wilcoxon
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'fm_learning_diagnostic/py'))
from diagnose import context,stats,PILOT,REPO,sha,save,matrix,fm_to_qubo,train_factorization_machine
from fm import TorchFM
P=json.loads((OUT/'json/protocol.json').read_text())

def configs():
 return [{'id':f'k{k}_{opt}_wd{wd}','rank':k,'optimizer':opt,'weight_decay':wd} for k in [1,2] for opt,wd in [('Adam',0.),('AdamW',0.),('AdamW',.01)]]

def state_hash(m):
 h=hashlib.sha256()
 for name,t in m.state_dict().items():h.update(name.encode());h.update(t.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def fit(c,s,x,y):
 torch.manual_seed(s);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 initial=TorchFM(23,c['rank']);init_hash=state_hash(initial)
 torch.manual_seed(s)
 m=train_factorization_machine(torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.float32),23,k=c['rank'],epochs=P['epochs'],learning_rate=P['learning_rate'],optimizer_name=c['optimizer'],weight_decay=c['weight_decay']);m.eval()
 return m,initial,init_hash

def prevalidate():
 assert P['status']=='ready'
 manifest=json.loads((ROOT/'json/problem_manifest.json').read_text());assert manifest['prevalidation']['legacy_n6_n9_passed'] and manifest['prevalidation']['target_group_sizes_passed']
 assert (REPO/'src/fm.py').read_bytes()==(REPO/'intern/src/fm.py').read_bytes()
 errs=[]
 for s in P['seeds']:
  old=json.loads((ROOT/f'fm_learning_diagnostic/json/seed_{s}.json').read_text());base=json.loads((PILOT/f'json/seed_{s}.json').read_text());_,x,y,tr,te=context(base)
  for c in configs():
   if c['optimizer']!='Adam':continue
   m,_,_=fit(c,s,x[tr],y[tr]);q,b=fm_to_qubo(m);pred=np.einsum('bi,ij,bj->b',x,matrix(q,23),x)+b
   f=next(f for f in old['fits'] if f['config']['id']==f"k{c['rank']}_lr0.1_e120_raw");err=float(np.max(np.abs(pred-np.array(f['predictions']))));assert err<1e-6;errs.append(err)
 # A zero feature column cannot get an MSE gradient. AdamW must decay its factor.
 z=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
 for opt,wd in [('Adam',0.),('AdamW',0.),('AdamW',.01)]:
  torch.manual_seed(42);init=TorchFM(3,1);torch.manual_seed(42)
  m=train_factorization_machine(z,torch.tensor([1.,2.]),3,k=1,epochs=3,learning_rate=.1,optimizer_name=opt,weight_decay=wd)
  ratio=(1-.1*wd)**3 if opt=='AdamW' else 1.
  torch.testing.assert_close(m.V[2],init.V[2]*ratio,rtol=1e-6,atol=1e-7)
 for kw in [{'optimizer_name':'invalid'},{'weight_decay':-1},{'weight_decay':float('nan')}]:
  try:train_factorization_machine(z,torch.tensor([1.,2.]),3,**kw)
  except ValueError:pass
  else:raise AssertionError('invalid option accepted')
 save(OUT/'json/prevalidation.json',{'status':'passed','runner_sha256':sha(Path(__file__)),'shared_fm_sha256':sha(REPO/'src/fm.py'),'baseline_reproduction_10_max_errors':errs,'zero_feature_decay_and_invalid_options':'passed','inherited_N6_N9_23bit_circuit_checks':'passed; no circuit changes'})
 print('prevalidation passed',flush=True)

def run():
 v=json.loads((OUT/'json/prevalidation.json').read_text());assert v['runner_sha256']==sha(Path(__file__)) and v['shared_fm_sha256']==sha(REPO/'src/fm.py')
 for s in P['seeds']:
  target=OUT/f'json/seed_{s}.json';assert not target.exists()
  base=json.loads((PILOT/f'json/seed_{s}.json').read_text());cats,x,y,tr,te=context(base);seen={cats[i][0] for i in tr};missing=[j for j in range(16) if np.all(x[tr,j]==0)]
  d={'seed':s,'status':'running','initial_sha256':base['initial_sha256'],'source_sha256':sha(PILOT/f'json/seed_{s}.json'),'train_ids':tr,'test_ids':te,'missing_organic_indices':missing,'fits':[]};save(target,d)
  for c in configs():
   start=time.perf_counter();m,initial,ih=fit(c,s,x[tr],y[tr]);qd,b=fm_to_qubo(m);q=matrix(qd,23);pred=np.einsum('bi,ij,bj->b',x,q,x)+b
   with torch.no_grad():direct=m(torch.tensor(x,dtype=torch.float32)).numpy()
   err=float(np.max(np.abs(pred-direct)));assert err<1e-4
   chosen=min(te,key=lambda i:(pred[i],tuple(x[i])))
   metrics={'train_rmse':float(np.sqrt(np.mean((pred[tr]-y[tr])**2))),'test_rmse':float(np.sqrt(np.mean((pred[te]-y[te])**2))),'test_spearman':float(spearmanr(pred[te],y[te]).statistic),'selected_true_value':float(y[chosen]),'selected_test_regret':float((y[chosen]-y[te].min())/(y[te].max()-y[te].min()))}
   for label,ids in [('seen_organic',[i for i in te if cats[i][0] in seen]),('unseen_organic',[i for i in te if cats[i][0] not in seen])]:metrics[label+'_rmse']=float(np.sqrt(np.mean((pred[ids]-y[ids])**2)))
   expected=(1-P['learning_rate']*c['weight_decay'])**P['epochs'] if c['optimizer']=='AdamW' else 1.
   decayerr=float(torch.max(torch.abs(m.V[missing]-initial.V[missing]*expected)).detach());assert decayerr<1e-5
   cp=OUT/f"data/checkpoints/seed{s}_{c['id']}.pt";torch.save({'state_dict':m.state_dict(),'config':c,'initial_state_sha256':ih,'initial_dataset_sha256':d['initial_sha256']},cp)
   d['fits'].append({'config':c,'initial_state_sha256':ih,'metrics':metrics,'selected_id':chosen,'selected_categories':cats[chosen],'predictions':pred.tolist(),'qubo':q.tolist(),'offset':b,'parity_max_error':err,'expected_unseen_factor_ratio':expected,'unseen_factor_decay_max_error':decayerr,'checkpoint':str(cp.relative_to(OUT)),'checkpoint_sha256':sha(cp),'seconds':time.perf_counter()-start});save(target,d)
  d['status']='completed';save(target,d);print('completed seed',s,flush=True)
 save(OUT/'json/environment.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'threads':1,'device':'CPU','protocol_sha256':sha(OUT/'json/protocol.json'),'runner_sha256':sha(Path(__file__)),'shared_fm_sha256':sha(REPO/'src/fm.py'),'conversion_sha256':sha(REPO/'src/fm_to_qubo.py')})

def analyze():
 rows=[json.loads((OUT/f'json/seed_{s}.json').read_text()) for s in P['seeds']];assert all(d['status']=='completed' and len(d['fits'])==6 for d in rows)
 groups=[];tests=[]
 for d in rows:
  base=json.loads((PILOT/f"json/seed_{d['seed']}.json").read_text());_,x,y,tr,te=context(base);assert sha(PILOT/f"json/seed_{d['seed']}.json")==d['source_sha256'] and d['initial_sha256']==base['initial_sha256']
  for k in [1,2]:assert len({f['initial_state_sha256'] for f in d['fits'] if f['config']['rank']==k})==1
  for f in d['fits']:
   assert sha(OUT/f['checkpoint'])==f['checkpoint_sha256'];pred=np.array(f['predictions']);np.testing.assert_allclose(pred,np.einsum('bi,ij,bj->b',x,np.array(f['qubo']),x)+f['offset'],atol=1e-10)
   m=TorchFM(23,f['config']['rank']);m.load_state_dict(torch.load(OUT/f['checkpoint'],weights_only=False)['state_dict'])
   with torch.no_grad():np.testing.assert_allclose(pred,m(torch.tensor(x,dtype=torch.float32)).numpy(),atol=1e-4,rtol=0)
   assert f['selected_id']==min(te,key=lambda i:(pred[i],tuple(x[i])))
   assert abs(f['metrics']['test_rmse']-np.sqrt(np.mean((pred[te]-y[te])**2)))<1e-10
 for c in configs():
  fs=[next(f for f in d['fits'] if f['config']['id']==c['id']) for d in rows];groups.append({'config':c,'statistics':{key:stats([f['metrics'][key] for f in fs]) for key in fs[0]['metrics']}})
 for k in [1,2]:
  for a,b in [('AdamW_wd0.0','Adam_wd0.0'),('AdamW_wd0.01','AdamW_wd0.0'),('AdamW_wd0.01','Adam_wd0.0')]:
   for key in ['test_rmse','test_spearman','selected_true_value']:
    ds=[next(f for f in d['fits'] if f['config']['id']==f'k{k}_{a}')['metrics'][key]-next(f for f in d['fits'] if f['config']['id']==f'k{k}_{b}')['metrics'][key] for d in rows];p=float(wilcoxon(ds).pvalue) if np.any(ds) else 1.
    tests.append({'rank':k,'contrast':a+' minus '+b,'metric':key,'paired_differences':ds,'median_difference':float(np.median(ds)),'p_raw':p,'p_bonferroni':min(1.,p*18)})
 save(OUT/'json/summary.json',{'status':'completed','fits':30,'groups':groups,'paired_tests':tests});save(OUT/'json/artifact_verification.json',{'status':'passed','fits':30,'checks':'same dataset/init per rank-seed; 30 checkpoint hashes and forward/QUBO parity; independently recomputed test RMSE and exact predicted selection; no dropped fits'})
 print(json.dumps(groups,indent=2),flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
