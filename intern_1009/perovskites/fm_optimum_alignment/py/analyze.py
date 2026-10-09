"""Offline FM-vs-truth argmin audit of saved initial3 trajectories; no training."""
import sys,json,argparse
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wilcoxon
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OUT.parent/'xy_p3_initial3_bbo/py'))
from analyze_initial3 import OUT as SOURCE,P,CODE,REPO,problem,sha,save,checked,TorchFM,trip,plt

def compute():
 protocol=json.loads((OUT/'json/protocol.json').read_text());assert protocol['status']=='ready'
 v=json.loads((SOURCE/'json/prevalidation.json').read_text());assert v['status']=='passed'
 assert v['source_sha256']=={str(c.relative_to(REPO)):sha(c) for c in CODE}
 assert sha(SOURCE/'json/protocol.json')==protocol['source_protocol_sha256']
 bb,cats,x=problem();assert bb.data_sha256==P['data_sha256'];truth=np.array([bb.evaluate(c) for c in cats]);trueids=np.flatnonzero(np.isclose(truth,truth.min(),atol=1e-10,rtol=0));rows=[];sources=[]
 for seed in P['seeds']:
  for method in P['methods']:
   path=SOURCE/f'json/{method}_seed{seed}.json';d=json.loads(path.read_text());assert d['status']=='completed' and len(d['events'])==77
   assert d['protocol_sha256']==protocol['source_protocol_sha256'] and sha(SOURCE/d['checkpoint'])==d['checkpoint_sha256'];states=torch.load(SOURCE/d['checkpoint'],weights_only=False);assert len(states)==77
   sources.append({'path':str(path.relative_to(OUT.parent)),'sha256':sha(path),'checkpoint_sha256':d['checkpoint_sha256']})
   for e,cp in zip(d['events'],states):
    m=TorchFM(23,1);m.load_state_dict(cp['state_dict']);checked(e,m,x,e['train_ids_before'],truth)
    pred=np.array(e['fm_predictions']);assert pred.shape==(192,) and np.isfinite(pred).all();ids=np.arange(192);unseen=ids[~np.isin(ids,e['train_ids_before'])]
    def mins(pool):return pool[np.isclose(pred[pool],min(pred[pool]),atol=1e-8,rtol=0)].tolist()
    globalids=mins(ids);unseenids=mins(unseen);chosen=min(globalids,key=lambda z:tuple(cats[z]));chosenunseen=min(unseenids,key=lambda z:tuple(cats[z]));match=bool(set(globalids)&set(trueids));pending=bool(set(trueids)&set(unseen))
    rows.append({'method':method,'seed':seed,'cycle':e['cycle'],'evaluations_before':e['evaluations_before'],'model_sha256':e['model_sha256'],'global_fm_min_ids':globalids,'global_match':match,'global_deterministic_id':int(chosen),'global_chosen_true_value':float(truth[chosen]),'global_chosen_true_regret':float((truth[chosen]-truth.min())/(truth.max()-truth.min())),'global_all_ties_true_optimal':all(z in trueids for z in globalids),'true_opt_best_FM_rank':int(1+np.sum(pred<min(pred[trueids])-1e-8)),'true_opt_FM_gap':float(min(pred[trueids])-min(pred)),'true_opt_still_unseen':pending,'unseen_fm_min_ids':unseenids,'unseen_match_when_pending':bool(set(unseenids)&set(trueids)) if pending else None,'unseen_deterministic_id':int(chosenunseen),'unseen_chosen_true_value':float(truth[chosenunseen]),'selected_id':e['selected_id'],'selected_hits_unseen_fm_min':e['selected_id'] in unseenids})
 assert len(rows)==1155
 for seed in P['seeds']:
  first=[r for r in rows if r['seed']==seed and r['cycle']==1];assert len({r['model_sha256'] for r in first})==1
 runs=[]
 for seed in P['seeds']:
  for method in P['methods']:
   rr=[r for r in rows if r['seed']==seed and r['method']==method];pending=[r for r in rr if r['true_opt_still_unseen']]
   runs.append({'seed':seed,'method':method,'global_match_fraction':float(np.mean([r['global_match'] for r in rr])),'global_chosen_true_regret_median':float(np.median([r['global_chosen_true_regret'] for r in rr])),'pending_models':len(pending),'unseen_match_before_true_hit':float(np.mean([r['unseen_match_when_pending'] for r in pending])) if pending else None})
 tests=[]
 for reference in ['XY-p1','LargePenalty-alpha1000']:
  for key in ['global_match_fraction','global_chosen_true_regret_median']:
   diffs=[next(r for r in runs if r['seed']==s and r['method']=='XY-p3')[key]-next(r for r in runs if r['seed']==s and r['method']==reference)[key] for s in P['seeds']];pv=float(wilcoxon(diffs).pvalue) if any(abs(z)>1e-12 for z in diffs) else 1.;tests.append({'reference':reference,'metric':key,'paired_differences':diffs,'p_raw':pv,'p_bonferroni':min(1,4*pv)})
 save(OUT/'json/summary.json',{'status':'completed','model_count':len(rows),'true_opt_ids':trueids.tolist(),'true_opt_candidates':[list(cats[i]) for i in trueids],'true_opt_value':float(truth.min()),'rows':rows,'runs':runs,'paired_tests':tests,'sources':sources,'protocol_sha256':sha(OUT/'json/protocol.json')})
 print('Audited 1155 saved FM models; summary saved.')

def plot():
 s=json.loads((OUT/'json/summary.json').read_text());assert s['status']=='completed' and s['protocol_sha256']==sha(OUT/'json/protocol.json');files=[]
 specs=[('global_match','FM argmin contains true optimum'),('global_chosen_true_value','True objective at FM argmin (eV)'),('true_opt_best_FM_rank','Best FM rank of true optimum'),('true_opt_FM_gap','FM gap of true optimum (eV)')];colors=['#2ca02c','#008080','#ff7f0e'];styles=['-',':','--']
 def draw(ax,key,label):
  for method,col,style in zip(P['methods'],colors,styles):
   a=np.array([[r[key] for r in s['rows'] if r['method']==method and r['seed']==seed] for seed in P['seeds']],float);lo,med,hi=np.percentile(a,[25,50,75],axis=0);domain=np.arange(1,78);ax.plot(domain,med,color=col,linestyle=style,label=method);ax.fill_between(domain,lo,hi,color=col,alpha=.12)
  ax.set_xlabel('FM training cycle (before candidate selection)');ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=8)
  if key=='global_match':ax.set_ylim(-.05,1.05)
 def export(fig,name):
  for ext in ['pdf','svg','png']:
   path=OUT/ext/f'{name}.{ext}';fig.savefig(path,dpi=250,bbox_inches='tight');files.append({'path':str(path.relative_to(OUT)),'sha256':sha(path)})
 fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
 for ax,(key,label) in zip(axes.flat,specs):draw(ax,key,label)
 export(fig,'FM_optimum_alignment');plt.close(fig)
 for key,label in specs:
  fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained');draw(ax,key,label);export(fig,key);plt.close(fig)
 lines=['# FM最小解と実最適解の一致診断','','保存済み初期3件BBOの15Run・1155学習済みFMを全192実行可能候補で評価。再学習・追加BBOなし。ペナルティなしのFM値を比較。各サイクルは候補採用前。','','真の最適値: '+str(s['true_opt_value'])+' eV。材料: '+str(s['true_opt_candidates']), '', '| 手法 | 全77モデルの一致率・5Seed中央値[IQR] | FM最小解の真regret・Run中央値を5Seed集計 | 発見前の未評価FM最小解一致率 |','|---|---:|---:|---:|']
 for method in P['methods']:
  rr=[r for r in s['runs'] if r['method']==method];cells=[]
  for key in ['global_match_fraction','global_chosen_true_regret_median','unseen_match_before_true_hit']:
   t=trip([r[key] for r in rr if r[key] is not None]);cells.append(f"{t['median']:.6g} [{t['q1']:.6g}, {t['q3']:.6g}]")
  lines.append('| '+method+' | '+' | '.join(cells)+' |')
 lines+=['','初期FMは3手法で共通。以降は探索履歴と学習集合が異なるため、手法差をサンプラー単独の因果効果としない。77サイクルを独立1155試行とみなさず、統計単位は5Seed。全空間FM最小解には既評価候補も含める。真の最適解が評価済みなら、未評価候補での一致指標は欠測とし、失敗と数えない。FM同率最小は絶対許容1e-8、真最小は1e-10。集合の交差と決定的な材料tuple順の最小解を別々に記録。学習後の全体一致だけで探索時の有用性を結論しない。','','## 初期3件のFM（5個、手法間共通）','','| Seed | FM最小材料 | 真値 eV | 一致 | 真最適解のFM順位 |','|---:|---|---:|---|---:|']
 bb,cats,x=problem()
 for r in s['rows']:
  if r['method']=='XY-p3' and r['cycle']==1:lines.append(f"| {r['seed']} | {cats[r['global_deterministic_id']]} | {r['global_chosen_true_value']:.6g} | {r['global_match']} | {r['true_opt_best_FM_rank']} |")
 lines+=['','## 対応検定（4比較Bonferroni、非有意は同等性の証明ではない）','','| 対照 | 指標 | 補正p |','|---|---|---:|']
 for t in s['paired_tests']:lines.append(f"| {t['reference']} | {t['metric']} | {t['p_bonferroni']:.6g} |")
 (OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n');save(OUT/'json/figure_manifest.json',{'summary_sha256':sha(OUT/'json/summary.json'),'files':files,'visual_QA':'pending next interactive review'});print('Report and 15 PDF/SVG/PNG figures saved; visual review pending.')

if __name__=='__main__':
 torch.set_num_threads(1);ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['compute','plot']);a=ap.parse_args();compute() if a.stage=='compute' else plot()
