"""Same-data exploratory FM rank/training/scaling diagnostic; shared FM implementation."""
import sys,json,time,itertools,platform,importlib.metadata
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr,wilcoxon
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent;PILOT=ROOT/'fixed_fm_pilot'
sys.path.insert(0,str(PILOT/'py'))
from run_pilot import PerovskitesEvaluator,train_factorization_machine,fm_to_qubo,matrix,sha,save,REPO
P=json.loads((OUT/'json/protocol.json').read_text());BASE='k2_lr0.1_e120_raw'

def configs():
 rows=[{'rank':k,'lr':lr,'epochs':ep,'scale':'raw'} for k,lr,ep in itertools.product(P['grid']['ranks'],P['grid']['learning_rates'],P['grid']['epochs'])]
 rows +=[{'rank':r['rank'],'lr':r['learning_rate'],'epochs':r['epochs'],'scale':'standardized'} for r in P['additional_standardized_configs']]
 for r in rows:r['id']=f"k{r['rank']}_lr{r['lr']}_e{r['epochs']}_{r['scale']}"
 return rows

def fit(cfg,seed,x,y):
 torch.manual_seed(seed);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 mu=float(y.mean()) if cfg['scale']=='standardized' else 0.;sigma=float(y.std()) if cfg['scale']=='standardized' else 1.;assert sigma>0
 m=train_factorization_machine(torch.tensor(x,dtype=torch.float32),torch.tensor((y-mu)/sigma,dtype=torch.float32),23,k=cfg['rank'],epochs=cfg['epochs'],learning_rate=cfg['lr']);m.eval()
 return m,mu,sigma

def context(base):
 bb=PerovskitesEvaluator();cats=[list(c) for c in bb.candidates()];x=np.array([bb.encode(c) for c in cats]);y=np.array([bb.evaluate(c) for c in cats]);train=base['initial_dataset']['candidate_ids'];test=[i for i in range(192) if i not in train];return cats,x,y,train,test

def prevalidate():
 assert P['status']=='ready' and json.loads((PILOT/'json/artifact_verification.json').read_text())['status']=='passed'
 manifest=json.loads((ROOT/'json/problem_manifest.json').read_text());assert manifest['prevalidation']['legacy_n6_n9_passed'] and manifest['prevalidation']['target_group_sizes_passed']
 cfg=next(c for c in configs() if c['id']==BASE);errs=[]
 for seed in P['seeds']:
  base=json.loads((PILOT/f'json/seed_{seed}.json').read_text());cats,x,y,tr,te=context(base);assert sha(PILOT/base['checkpoint'])==base['checkpoint_sha256']
  m,mu,sigma=fit(cfg,seed,x[tr],y[tr]);qd,bias=fm_to_qubo(m);q=matrix(qd,23);pred=np.einsum('bi,ij,bj->b',x,q,x)+bias
  old=np.array(base['fm_quality']['all_candidate_predictions']);error=float(np.max(np.abs(pred-old)));assert error<1e-6,error;errs.append(error)
 save(OUT/'json/prevalidation.json',{'status':'passed','runner_sha256':sha(Path(__file__)),'baseline_reproduction_max_errors':errs,'source_N6_N9_and23_validation':'passed'})
 print('prevalidated baseline reproduction',errs,flush=True)

def run():
 assert json.loads((OUT/'json/prevalidation.json').read_text())['runner_sha256']==sha(Path(__file__))
 save(OUT/'json/environment.json',{'python':sys.version,'platform':platform.platform(),'device':'CPU','threads':1,'versions':{n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','matplotlib']},'protocol_sha256':sha(OUT/'json/protocol.json'),'code_sha256':{str(p.relative_to(REPO)):sha(p) for p in [Path(__file__),REPO/'src/fm.py',REPO/'src/fm_to_qubo.py']},'CPU_model_and_peakRSS':'unmeasured'})
 for seed in P['seeds']:
  target=OUT/f'json/seed_{seed}.json';assert not target.exists(),'preserve results'
  base=json.loads((PILOT/f'json/seed_{seed}.json').read_text());cats,x,y,tr,te=context(base)
  seenorg={cats[i][0] for i in tr};missing=sorted({c[0] for c in cats}-seenorg)
  d={'seed':seed,'status':'running','source_sha256':sha(PILOT/f'json/seed_{seed}.json'),'initial_sha256':base['initial_sha256'],'train_ids':tr,'test_ids':te,'organic_coverage':len(seenorg),'missing_organic':missing,'fits':[]};save(target,d)
  for cfg in configs():
   start=time.perf_counter();m,mu,sigma=fit(cfg,seed,x[tr],y[tr]);qd,bias=fm_to_qubo(m);q=matrix(qd,23)*sigma;offset=bias*sigma+mu
   pred=np.einsum('bi,ij,bj->b',x,q,x)+offset
   with torch.no_grad():direct=m(torch.tensor(x,dtype=torch.float32)).numpy()*sigma+mu
   error=float(np.max(np.abs(pred-direct)));assert error<1e-4,error
   chosen=min(te,key=lambda i:(pred[i],tuple(x[i])));best_true=float(y[te].min());den=float(y[te].max()-best_true)
   metrics={'train_rmse':float(np.sqrt(np.mean((pred[tr]-y[tr])**2))),'test_rmse':float(np.sqrt(np.mean((pred[te]-y[te])**2))),'test_spearman':float(spearmanr(pred[te],y[te]).statistic),'selected_true_value':float(y[chosen]),'selected_test_regret':float((y[chosen]-best_true)/den),'test_true_minimum':best_true,'negative_predictions':int(np.sum(pred[te]<0))}
   for label,ids in [('seen_organic',[i for i in te if cats[i][0] in seenorg]),('unseen_organic',[i for i in te if cats[i][0] not in seenorg])]:
    metrics[f'{label}_rmse']=float(np.sqrt(np.mean((pred[ids]-y[ids])**2))) if ids else None;metrics[f'{label}_n']=len(ids)
   cp=OUT/f"data/checkpoints/seed{seed}_{cfg['id']}.pt";torch.save({'state_dict':m.state_dict(),'config':cfg,'target_mean':mu,'target_std':sigma,'initial_sha256':base['initial_sha256']},cp)
   d['fits'].append({'config':cfg,'metrics':metrics,'selected_id':chosen,'selected_categories':cats[chosen],'selected_prediction':float(pred[chosen]),'predictions':pred.tolist(),'qubo':q.tolist(),'offset':offset,'parity_max_error':error,'seconds':time.perf_counter()-start,'checkpoint':str(cp.relative_to(OUT)),'checkpoint_sha256':sha(cp)});save(target,d)
  d['status']='completed';save(target,d);print('completed seed',seed,flush=True)

def stats(vals):return {k:float(v) for k,v in zip(['median','q1','q3'],[np.median(vals),np.quantile(vals,.25),np.quantile(vals,.75)])}

def analyze():
 allseeds=[json.loads((OUT/f'json/seed_{s}.json').read_text()) for s in P['seeds']];assert all(d['status']=='completed' and len(d['fits'])==18 for d in allseeds)
 groups=[];tests=[];metrics=['train_rmse','test_rmse','test_spearman','selected_true_value','selected_test_regret','seen_organic_rmse','unseen_organic_rmse']
 for cfg in configs():
  fits=[next(f for f in d['fits'] if f['config']['id']==cfg['id']) for d in allseeds]
  for d,f in zip(allseeds,fits):
   assert d['source_sha256']==sha(PILOT/f"json/seed_{d['seed']}.json") and sha(OUT/f['checkpoint'])==f['checkpoint_sha256']
   base=json.loads((PILOT/f"json/seed_{d['seed']}.json").read_text());cats,x,y,tr,te=context(base);pred=np.array(f['predictions']);q=np.array(f['qubo']);np.testing.assert_allclose(pred,np.einsum('bi,ij,bj->b',x,q,x)+f['offset'],atol=1e-10)
   assert f['selected_id']==min(te,key=lambda i:(pred[i],tuple(x[i])))
   assert abs(f['metrics']['test_rmse']-np.sqrt(np.mean((pred[te]-y[te])**2)))<1e-10
  groups.append({'config':cfg,'statistics':{k:stats([f['metrics'][k] for f in fits]) for k in metrics}})
  if cfg['id']!=BASE:
   for key in ['test_rmse','test_spearman','selected_true_value']:
    diffs=[next(f for f in d['fits'] if f['config']['id']==cfg['id'])['metrics'][key]-next(f for f in d['fits'] if f['config']['id']==BASE)['metrics'][key] for d in allseeds];p=float(wilcoxon(diffs).pvalue) if np.any(diffs) else 1.
    tests.append({'configuration':cfg['id'],'metric':key,'differences':diffs,'median_difference':float(np.median(diffs)),'p_raw':p,'p_bonferroni':min(1.,p*51)})
 result={'status':'completed','model_fits':90,'groups':groups,'paired_tests':tests};save(OUT/'json/summary.json',result);save(OUT/'json/artifact_verification.json',{'status':'passed','fits':90,'checks':'complete18config x5seeds,source/checkpoint SHA,independent QUBO predictions/ranking/RMSE parity; perfit Torch/QUBO parity'})
 lines=['# FM学習改善の初期診断','','同じ初期20件、5Seed、18条件・90学習。全て共通172候補を診断に使う。数値は中央値 [Q1,Q3]。testを見た探索的比較で、選んだ設定の未使用test評価ではない。','','| 設定 | Train RMSE eV | Test RMSE eV | Test Spearman | 予測最小未評価候補の真値eV |','|---|---:|---:|---:|---:|']
 for g in groups:
  fmt=lambda k:'{median:.4g} [{q1:.4g},{q3:.4g}]'.format(**g['statistics'][k]);lines.append('| '+g['config']['id']+' | '+' | '.join(fmt(k) for k in ['train_rmse','test_rmse','test_spearman','selected_true_value'])+' |')
 lines+=['','k0は相互作用なし。rawは生eV、standardizedは初期20件だけからmean/std算出。予測値はeVへ戻す。未評価候補の予測最小を全列挙で選び、真値で順位付けしない。学習回数を増やす実験はBBOサイクル追加ではない。正則化・初期集合変更・独立学習乱数の反復は未実施。']
 (OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n');plot(result);print('verified90 fits',flush=True)

def plot(result):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 # Rank effect at original lr/epochs; other factors retained in complete table.
 fig,axs=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
 for ax,key,label in zip(axs,['test_rmse','test_spearman','selected_true_value'],['Test RMSE (eV)','Test Spearman','True value of predicted best unseen (eV)']):
  for lr,ep,color,marker in [(0.1,120,'#1f77b4','o'),(0.01,1000,'#ff7f0e','s')]:
   gs=[next(g for g in result['groups'] if g['config']['id']==f'k{k}_lr{lr}_e{ep}_raw')['statistics'][key] for k in [0,1,2,4]]
   val=lambda name:np.array([g[name] for g in gs]);ax.plot([0,1,2,4],val('median'),color=color,marker=marker,label=f'lr={lr}, epochs={ep}');ax.fill_between([0,1,2,4],val('q1'),val('q3'),color=color,alpha=.12)
  ax.set_xticks([0,1,2,4]);ax.set_xlabel('FM rank (0: additive)');ax.set_ylabel(label);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
 for ext in ['pdf','png','svg']:fig.savefig(OUT/f'{ext}/fm_learning_overview.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)

if __name__=='__main__':globals()[sys.argv[1]]()
