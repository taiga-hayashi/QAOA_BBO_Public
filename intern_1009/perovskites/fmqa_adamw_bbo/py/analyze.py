"""Independent trajectory/checkpoint audit, predeclared statistics and plots."""
import sys,json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wilcoxon
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OUT/'py'))
from run_bbo import P,ROOT,REPO,PerovskitesEvaluator,save,sha,select,model_hash,matrix,fm_to_qubo
from fm import TorchFM

def stat(a):return {k:float(v) for k,v in zip(['median','q1','q3'],np.quantile(a,[.5,.25,.75]))}

def main():
 bb=PerovskitesEvaluator();cats=list(bb.candidates());x=np.array([bb.encode(c) for c in cats]);truth=np.array([bb.evaluate(c) for c in cats]);fopt=float(truth.min());fworst=float(truth.max());rows=[];qualities=[]
 torch.set_num_threads(1)
 for cfg in P['fm_settings']:
  for method in P['methods']:
   for seed in P['seeds']:
    path=OUT/f"json/{cfg['id']}_{method}_seed{seed}.json";d=json.loads(path.read_text());assert d['status']=='completed' and len(d['events'])==60
    basepath=ROOT/f'fixed_fm_pilot/json/seed_{seed}.json';base=json.loads(basepath.read_text());assert d['source_file_sha256']==sha(basepath) and d['source_initial_sha256']==base['initial_sha256'] and d['initial_ids']==base['initial_dataset']['candidate_ids']
    assert d['protocol_sha256']==sha(OUT/'json/protocol.json') and sha(OUT/d['checkpoint'])==d['checkpoint_sha256']
    states=torch.load(OUT/d['checkpoint'],weights_only=False);assert len(states)==60
    ids=list(d['initial_ids']);vals=list(d['initial_values']);np.testing.assert_allclose(vals,truth[ids],atol=0,rtol=0);best=min(vals);actual=len(ids);alpha=1.
    for e,cp in zip(d['events'],states):
     assert e['train_ids_before']==ids and e['evaluations_before']==actual and e['cycle']==cp['cycle']
     m=TorchFM(23,cfg['rank']);m.load_state_dict(cp['state_dict']);assert model_hash(m)==cp['model_sha256']==e['model_sha256'];pred=np.array(e['fm_predictions'])
     qd,b=fm_to_qubo(m);q=matrix(qd,23);np.testing.assert_allclose(pred,np.einsum('bi,ij,bj->b',x,q,x)+b,atol=1e-10,rtol=0)
     with torch.no_grad():np.testing.assert_allclose(pred,m(torch.tensor(x,dtype=torch.float32)).numpy(),atol=1e-4,rtol=0)
     chosen,meta=select(np.array(e['raw_bits']),pred,x,set(ids));assert chosen==e['selected_id'] and all(meta[k]==e[k] for k in meta) and len(e['raw_bits'])==10
     assert abs(e['lambda']-e['base_Ising_scale_S']*e['alpha'])<1e-10
     if method=='Adaptive-FMQA':
      assert abs(e['alpha']-alpha)<1e-12;r=e['raw_feasible_rate'];alpha=min(100.,alpha*2) if r<.7 else min(100.,alpha*1.25) if r<1 else max(.01,alpha*.85);assert abs(e['next_alpha']-alpha)<1e-12
     else:assert e['alpha']==e['next_alpha']==100.
     te=[i for i in range(192) if i not in ids];qualities.append({'setting':cfg['id'],'method':method,'seed':seed,'cycle':e['cycle'],'test_rmse':float(np.sqrt(np.mean((pred[te]-truth[te])**2)))})
     if chosen is not None:
      assert chosen not in ids and e['selected_value']==truth[chosen] and abs(e['selected_prediction']-pred[chosen])<1e-12;ids.append(chosen);vals.append(e['selected_value']);actual+=1;best=min(best,e['selected_value'])
     else:assert e['selected_value'] is None
     assert e['evaluations_after']==actual and e['best_so_far']==best and actual==len(set(ids))<=80
    assert d['final_ids']==ids and d['final_values']==vals and d['actual_evaluations']==actual and d['final_best']==best
    d['initial_optimum']=min(d['initial_values'])==fopt;d['newly_found_optimum']=not d['initial_optimum'] and best==fopt;d['first_optimum_evaluations']=20 if d['initial_optimum'] else next((e['evaluations_after'] for e in d['events'] if e['best_so_far']==fopt),None);d['final_regret']=(best-fopt)/(fworst-fopt);d['zero_candidate_cycles']=sum(e['selected_id'] is None for e in d['events']);rows.append(d)
 groups=[]
 for cfg in P['fm_settings']:
  for method in P['methods']:
   ds=[d for d in rows if d['fm_setting']['id']==cfg['id'] and d['method']==method]
   keys={'final_best':[d['final_best'] for d in ds],'final_regret':[d['final_regret'] for d in ds],'actual_evaluations':[d['actual_evaluations'] for d in ds],'zero_candidate_cycles':[d['zero_candidate_cycles'] for d in ds],'raw_feasible_rate':[np.mean([e['raw_feasible_rate'] for e in d['events']]) for d in ds],'seconds':[d['total_seconds'] for d in ds]}
   groups.append({'setting':cfg['id'],'method':method,'statistics':{k:stat(v) for k,v in keys.items()},'final_optimum_runs':sum(d['final_best']==fopt for d in ds),'initial_optimum_runs':sum(d['initial_optimum'] for d in ds),'newly_found_optimum_runs':sum(d['newly_found_optimum'] for d in ds),'per_seed':[{'seed':d['seed'],'initial_best':min(d['initial_values']),'final_best':d['final_best'],'actual_evaluations':d['actual_evaluations'],'zero_candidate_cycles':d['zero_candidate_cycles'],'initial_optimum':d['initial_optimum'],'newly_found_optimum':d['newly_found_optimum'],'first_optimum_evaluations':d['first_optimum_evaluations']} for d in ds]})
 tests=[]
 for method in P['methods']:
  for key in ['final_regret','actual_evaluations']:
   diffs=[next(d for d in rows if d['seed']==seed and d['method']==method and d['fm_setting']['id']=='proposed')[key]-next(d for d in rows if d['seed']==seed and d['method']==method and d['fm_setting']['id']=='original')[key] for seed in P['seeds']];p=float(wilcoxon(diffs).pvalue) if np.any(diffs) else 1.;tests.append({'method':method,'metric':key,'contrast':'proposed minus original','paired_differences':diffs,'median_difference':float(np.median(diffs)),'p_raw':p,'p_bonferroni':min(1.,4*p)})
 summary={'status':'completed','runs':20,'cycles':1200,'sa_samples':12000,'known_optimum':fopt,'known_worst':fworst,'groups':groups,'paired_tests':tests,'solver_evaluations_sum_including_initial_per_run':sum(d['actual_evaluations'] for d in rows),'failures':0}
 save(OUT/'json/summary.json',summary);save(OUT/'json/fm_quality_audit.json',{'scope':'offline all192 lookup truth, not training or proposal ranking or solver BB budget','rows':qualities});save(OUT/'json/artifact_verification.json',{'status':'passed','runs':20,'models_checked':1200,'checks':'initial/protocol/checkpoint hashes, all model hashes, Torch/QUBO parity, all raw sample filters and ranking, accumulated training ids and lookup values, budget accounting, adaptive alpha and lambda; all runs retained'})
 report(summary);plot(rows);print(json.dumps(summary,ensure_ascii=False,indent=2))

def report(s):
 lines=['# 再学習FMQAの結果','','23 One-Hot変数のPerovskites固定lookupで、2つのFM設定 ×2つのSA手法 ×5 Seedの20 Run、各60サイクルを完了。初期20件、10reads/cycle、最大1件採用、評価予算上限80。候補なしは0評価・追加サンプルなし。以下は5 Seedの中央値 [Q1,Q3]。','','| FM設定 | SA手法 | 最終best eV | Regret | 実評価数 | 候補なしサイクル | 最適解保持Run | 新規発見Run |','|---|---|---:|---:|---:|---:|---:|---:|']
 for g in s['groups']:
  fmt=lambda k:'{median:.4g} [{q1:.4g},{q3:.4g}]'.format(**g['statistics'][k]);lines.append('| '+g['setting']+' | '+g['method']+' | '+' | '.join(fmt(k) for k in ['final_best','final_regret','actual_evaluations','zero_candidate_cycles'])+f" | {g['final_optimum_runs']}/5 | {g['newly_found_optimum_runs']}/4 |")
 lines+=['','proposed＝rank1・AdamW・wd0.01、original＝rank2・Adam・wd0。両方lr0.1・120 epochs、生eV・fullbatch・毎サイクル新規初期化。初期20件は同じ元の集合で網羅設計に変えていない。Seed101は最初から真の最適解1.5249 eVを含むため、新規発見率の分母は残り4 Run。','','## Seed別結果','','| 設定 | 手法 | Seed | 初期best eV | 最終best eV | 実評価数 | 初期最適解 | 新規最適解 |','|---|---|---:|---:|---:|---:|---|---|']
 for g in s['groups']:
  for d in g['per_seed']:lines.append(f"| {g['setting']} | {g['method']} | {d['seed']} | {d['initial_best']:.4f} | {d['final_best']:.4f} | {d['actual_evaluations']} | {d['initial_optimum']} | {d['newly_found_optimum']} |")
 lines+=['','## ペア比較（提案−従来）','','| SA手法 | 指標 | 差分中央値 | raw p | Bonferroni p |','|---|---|---:|---:|---:|']
 for t in s['paired_tests']:lines.append(f"| {t['method']} | {t['metric']} | {t['median_difference']:.5g} | {t['p_raw']:.5g} | {t['p_bonferroni']:.5g} |")
 lines+=['','正規化Regretは全192候補のmin1.5249/max6.3242で計算。探索中に全真値は参照せず、訓練には初期値と採用候補の値のみ。全192真値を使うFM精度・最適性照合は別の事後診断。5 Seed、失敗・除外0。60サイクル打ち切りを収束と解釈しない。']
 (OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')

def plot(rows):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 def panel(ax,key):
  for method,color,marker,ls in [('Adaptive-FMQA','#1f77b4','o','-'),('LargePenalty-FMQA','#ff7f0e','s','--')]:
   for setting in ['proposed','original']:
    ds=[d for d in rows if d['method']==method and d['fm_setting']['id']==setting];label=method.replace('-FMQA','')+' / '+('rank1 AdamW' if setting=='proposed' else 'rank2 Adam');line=ls if setting=='proposed' else ':'
    if key=='best_evaluations':
     end=min(d['actual_evaluations'] for d in ds);xx=np.arange(20,end+1);ys=[]
     for d in ds:
      curve=[min(d['initial_values'])]+[e['best_so_far'] for e in d['events'] if e['selected_id'] is not None];ys.append(curve[:len(xx)])
    else:
     xx=np.arange(61) if key in ['best_cycles','evaluations_cycles'] else np.arange(1,61)
     ys=[[min(d['initial_values'])]+[e['best_so_far'] for e in d['events']] if key=='best_cycles' else [20]+[e['evaluations_after'] for e in d['events']] if key=='evaluations_cycles' else [e['raw_feasible_rate'] for e in d['events']] for d in ds]
    a=np.array(ys);q=np.quantile(a,[.25,.5,.75],axis=0);ax.plot(xx,q[1],color=color,linestyle=line,marker=marker,markevery=max(1,len(xx)//6),markersize=4,label=label);ax.fill_between(xx,q[0],q[2],color=color,alpha=.08)
  ax.set_xlabel('Actual BB evaluations' if key=='best_evaluations' else 'BBO cycle');ax.set_ylabel({'best_evaluations':'Best-so-far bandgap (eV)','best_cycles':'Best-so-far bandgap (eV)','evaluations_cycles':'Cumulative BB evaluations','feasibility_cycles':'Raw feasible fraction'}[key]);ax.grid(alpha=.2);ax.legend(fontsize=7)
  if key.startswith('best'):ax.axhline(1.5249,color='gray',linewidth=.8,linestyle='-.')
 keys=['best_evaluations','best_cycles','evaluations_cycles','feasibility_cycles'];fig,axs=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
 for ax,key in zip(axs.flat,keys):panel(ax,key)
 for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{ext}/fmqa_bbo_overview.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
 for key in keys:
  fig,ax=plt.subplots(figsize=(7,4.7),layout='constrained');panel(ax,key)
  for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{ext}/{key}.{ext}',dpi=300,bbox_inches='tight')
  plt.close(fig)
 save(OUT/'json/figure_manifest.json',{'files':[{'path':str(p.relative_to(OUT)),'sha256':sha(p)} for ext in ['pdf','svg','png'] for p in sorted((OUT/ext).glob('*.'+ext))],'scope':'median/IQR of5seeds; evaluation curves stop at minimum completed evaluations within each group, never extrapolated'})

if __name__=='__main__':main()
