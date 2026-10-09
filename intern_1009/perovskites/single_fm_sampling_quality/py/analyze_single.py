"""Reanalysis of measured batches for exactly one immutable FM."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent
sys.path.insert(0,str(ROOT/'large_penalty_diagnostic/py'))
from analyze_penalty import checked,TorchFM,torch,plt
from run_penalty import problem,sha,save,REPO,select,model_hash
P=json.loads((OUT/'json/protocol.json').read_text())

def stats(a):
 a=[float(v) for v in a if v is not None];return {'n':len(a),'median':float(np.median(a)) if a else None,'q1':float(np.percentile(a,25)) if a else None,'q3':float(np.percentile(a,75)) if a else None}

def main():
 torch.set_num_threads(1);assert P['status']=='ready';src=OUT/'data/source_fixed_seed42.json';cp=OUT/'data/fixed_model.pt';assert sha(src)==P['source_sha256'] and sha(REPO/P['source'])==P['source_sha256'];assert sha(cp)==P['checkpoint_sha256'];old=json.loads((ROOT/'large_penalty_diagnostic/json/prevalidation.json').read_text());assert old['status']=='passed'
 for f,h in old['source_sha256'].items():assert sha(REPO/f)==h
 d=json.loads(src.read_text());assert d['seed']==42 and d['status']=='completed';bb,cats,x=problem();truth=np.array([bb.evaluate(c) for c in cats]);ids=d['initial_ids'];assert len(ids)==10;np.testing.assert_array_equal(d['initial_values'],truth[ids]);m=TorchFM(23,1);m.load_state_dict(torch.load(cp,weights_only=True));pred=np.array(d['events'][0]['fm_predictions']);unseen=[i for i in range(192) if i not in ids];umin=float(pred[unseen].min());minimum=[i for i in unseen if abs(pred[i]-umin)<1e-8];lookup={tuple(row):i for i,row in enumerate(x)};groups=[];rows=[]
 for alpha in P['alphas']:
  es=sorted([e for e in d['events'] if e['schedule']=='auto' and e['alpha']==alpha],key=lambda e:e['rep']);assert len(es)==10 and [e['rep'] for e in es]==list(range(10));counts=np.zeros(192,dtype=int);batches=[]
  for e in es:
   checked(e,m,x,ids,truth);np.testing.assert_array_equal(e['fm_predictions'],pred);assert e['sampler_seed']==4200000+e['rep']*1000+17
   rawids=[lookup.get(tuple(row)) for row in e['raw_bits']];newids=[i for i in rawids if i is not None and i not in ids];gaps=pred[newids]-umin;np.add.at(counts,newids,1)
   row={'alpha':alpha,'rep':e['rep'],'feasible_rate':e['raw_feasible_rate'],'unseen_rate':len(newids)/100,'selected_gap':e['selected_unseen_fm_gap'],'selected_id':e['selected_id'],'mean_unseen_gap':float(gaps.mean()) if len(gaps) else None,'batch_min_capture':any(i in minimum for i in newids),'unique_unseen':len(set(newids)),'raw_candidate_ids':rawids,'beta_range':e['neal_info']['beta_range'],'lambda':e['lambda'],'S':e['base_Ising_scale_S']};batches.append(row);rows.append(row)
  n=int(counts.sum());poolids=np.repeat(np.arange(192),counts);poolgap=pred[poolids]-umin;prob=counts[counts>0]/n if n else np.array([]);g={'alpha':alpha,'raw_outputs':1000,'unseen_outputs':n,'raw_feasible_fraction':float(np.mean([e['raw_feasible_rate'] for e in es])),'unseen_fraction':n/1000,'pooled_mean_unseen_gap':float(poolgap.mean()) if n else None,'pooled_gap_stats':stats(poolgap),'selected_gap_stats':stats([r['selected_gap'] for r in batches]),'batch_min_capture_count':sum(bool(r['batch_min_capture']) for r in batches),'batch_count':10,'unique_unseen_pooled':int((counts>0).sum()),'batch_unique_stats':stats([r['unique_unseen'] for r in batches]),'effective_diversity':float(np.exp(-np.sum(prob*np.log(prob)))) if n else None,'empirical_unseen_frequencies':counts.tolist(),'pooled_unseen_gaps':poolgap.tolist(),'batch_selected_gaps':[r['selected_gap'] for r in batches]};groups.append(g)
 save(OUT/'json/prevalidation.json',{'status':'passed','single_model_sha256':model_hash(m),'source_hashes':'passed','raw_batches_verified':70,'FM_predictions_identical':'passed','FM_QUBO_and_candidate_filter_parity':'passed','inherited_N6_N9_N23':'passed; unchanged shared source hashes','source_sha256':old['source_sha256']})
 tests=[]
 for alpha in P['alphas']:
  if alpha==100:continue
  for metric in ['feasible_rate','selected_gap','mean_unseen_gap']:
   diffs=[];reps=[]
   for rep in range(10):
    a=next(r for r in rows if r['alpha']==alpha and r['rep']==rep)[metric];b=next(r for r in rows if r['alpha']==100 and r['rep']==rep)[metric]
    if a is not None and b is not None:diffs.append(float(a-b));reps.append(rep)
   p=float(wilcoxon(diffs,alternative='two-sided',method='auto').pvalue) if any(abs(v)>1e-12 for v in diffs) else 1.;tests.append({'alpha':alpha,'reference':100,'metric':metric,'sampling_repetitions':reps,'differences':diffs,'p_raw':p,'p_bonferroni':min(1.,18*p)})
 s={'status':'completed','model_seed':42,'model_count':1,'source_sha256':P['source_sha256'],'checkpoint_sha256':P['checkpoint_sha256'],'model_sha256':model_hash(m),'initial_ids':ids,'initial_values':d['initial_values'],'FM_predictions':pred.tolist(),'unseen_FM_minimum_value':umin,'unseen_FM_minimum_ids':minimum,'minimum_candidates':[{'id':i,'categories':list(cats[i]),'FM_value':float(pred[i]),'true_value_eV':float(truth[i])} for i in minimum],'groups':groups,'batches':rows,'paired_tests':tests};save(OUT/'json/summary.json',s);report(s);plot(s);print(json.dumps({'minimum':s['minimum_candidates'],'groups':[{k:g[k] for k in ['alpha','raw_feasible_fraction','pooled_mean_unseen_gap','selected_gap_stats','batch_min_capture_count','unique_unseen_pooled']} for g in groups],'significant_tests':[t for t in tests if t['p_bonferroni']<.05]},indent=2))

def report(s):
 lines=['# 1つの固定FMのサンプリング品質','','Seed42、初期10、rank1 AdamW wd0.01、lr0.1、120epochsの1モデルのみ。各alphaは保存済み100reads×10独立seed反復。新規学習・SA計算・BBOなし。自動beta、1000sweeps。モデル不変と全70batchesのraw選択を検証した。','','未評価182候補の全列挙FM最小値: '+str(s['unseen_FM_minimum_value'])+' eV。候補: '+json.dumps(s['minimum_candidates'],ensure_ascii=False)+'。真のBB最小値とは別。','','| α | raw制約率 | 未評価raw率 | raw未評価平均gap eV | 100reads選択gap中央値 [Q1,Q3] eV | FM最小捕捉 | 未評価ユニーク総数 |','|---:|---:|---:|---:|---:|---:|---:|']
 for g in s['groups']:
  t=g['selected_gap_stats'];text='未定義' if t['median'] is None else f"{t['median']:.6g} [{t['q1']:.6g},{t['q3']:.6g}]";lines.append(f"| {g['alpha']} | {g['raw_feasible_fraction']:.6g} | {g['unseen_fraction']:.6g} | {g['pooled_mean_unseen_gap']:.6g} | {text} | {g['batch_min_capture_count']}/10 | {g['unique_unseen_pooled']} |")
 lines+=['','raw平均gapは同じ候補の重複を含む、未評価実行可能サンプルの経験分布平均。選択gapは100readsを生成し既評価・違反を除去して最良1件を選んだ10反復の中央値/IQR。違反や候補なしは欠損として保持。最小値捕捉はgap<1e-8に相当する。ユニークが増えても良いFM値に集中しているとは限らない。','','## ペア検定','','同じ10sampling seedをalpha100と対応付け、3指標×6比較の18Wilcoxon、Bonferroni補正。このモデルに条件付けたサンプリングの比較であり、5FM比較やBBO性能の検定ではない。','','| α | 指標 | n | ペア差中央値 | 補正p |','|---:|---|---:|---:|---:|']
 for t in s['paired_tests']:lines.append(f"| {t['alpha']} | {t['metric']} | {len(t['differences'])} | {np.median(t['differences']):.6g} | {t['p_bonferroni']:.6g} |")
 (OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')

def plot(s):
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});gs=s['groups'];a=np.array(P['alphas']);colors=plt.get_cmap('tab10').colors;files=[]
 def draw(ax,key):
  if key=='gap_CDF':
   for g,c in zip(gs,colors):
    z=np.sort(g['pooled_unseen_gaps']);ax.step(z,np.arange(1,len(z)+1)/len(z),where='post',color=c,label=f"alpha={g['alpha']:g}")
   ax.set_xlabel('Unseen sample FM gap (eV)');ax.set_ylabel('Empirical cumulative fraction');ax.set_ylim(0,1.02)
  elif key=='FM_rank_frequency':
   ids=[i for i in range(192) if i not in s['initial_ids']];ids=sorted(ids,key=lambda i:(s['FM_predictions'][i],i))
   for g,c in zip(gs,colors):ax.plot(np.arange(1,183),np.array(g['empirical_unseen_frequencies'])[ids]/g['unseen_outputs'],color=c,label=f"alpha={g['alpha']:g}",alpha=.8)
   ax.set_xlabel('Exact unseen FM rank (1 = minimum)');ax.set_ylabel('Empirical unseen sample frequency')
  elif key=='feasibility':
   ax.plot(a,[g['raw_feasible_fraction'] for g in gs],'s--',color='#ff7f0e',label='Raw feasible');ax.plot(a,[g['unseen_fraction'] for g in gs],'o:',color='#9467bd',label='Raw unseen feasible');ax.set_ylabel('Raw sample fraction');ax.set_ylim(0,1.05)
  elif key=='minimum_capture':
   ax.plot(a,[g['batch_min_capture_count']/10 for g in gs],'s--',color='#ff7f0e',label='100-read batches, n=10');ax.set_ylabel('Exact unseen FM minimum capture');ax.set_ylim(0,1.05)
  elif key=='selected_gap':
   t=[g['selected_gap_stats'] for g in gs];med=np.array([v['median'] for v in t]);ax.errorbar(a,med,yerr=np.array([med-np.array([v['q1'] for v in t]),np.array([v['q3'] for v in t])-med]),fmt='s--',capsize=4,color='#ff7f0e',label='100-read best, median / IQR');ax.plot(a,[g['pooled_mean_unseen_gap'] for g in gs],'o:',color='#9467bd',label='Pooled raw unseen mean');ax.set_ylabel('FM gap to exact unseen minimum (eV)')
  else:
   ax.plot(a,[g['unique_unseen_pooled'] for g in gs],'s--',color='#ff7f0e',label='Unique unseen in 1,000 reads');ax.plot(a,[g['effective_diversity'] for g in gs],'o:',color='#9467bd',label='Effective unseen diversity');ax.set_ylabel('Candidate count')
  if key not in ['gap_CDF','FM_rank_frequency']:ax.set_xscale('log');ax.set_xlabel(r'Normalized penalty $\alpha=\lambda/S$')
  ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2 if key in ['gap_CDF','FM_rank_frequency'] else 1)
 def export(fig,key):
  for ext in ['pdf','svg','png']:
   p=OUT/ext/f'{key}.{ext}';fig.savefig(p,dpi=300,bbox_inches='tight');files.append({'path':str(p.relative_to(OUT)),'sha256':sha(p)})
 keys=['feasibility','selected_gap','minimum_capture','gap_CDF','FM_rank_frequency','diversity'];fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
 for ax,key in zip(axs.flat,keys):draw(ax,key)
 export(fig,'single_FM_sampling_overview');plt.close(fig)
 for key in keys:
  fig,ax=plt.subplots(figsize=(6.6,4.5),layout='constrained');draw(ax,key);export(fig,key);plt.close(fig)
 save(OUT/'json/figure_manifest.json',{'summary_sha256':sha(OUT/'json/summary.json'),'files':files})

if __name__=='__main__':main()
