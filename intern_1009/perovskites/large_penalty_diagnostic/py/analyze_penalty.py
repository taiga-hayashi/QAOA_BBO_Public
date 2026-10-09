"""Audit saved models/raw candidates, aggregate paired results and plot."""
import sys,json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_penalty import OUT,ROOT,REPO,P,CFG,CODE,initial,problem,select,sha,save,model_hash,matrix,build_penalty_bqm,qubo_to_ising_coefficients
from run_bbo import fm_to_qubo
from src.fm import TorchFM

def trip(values):
 a=[float(v) for v in values if v is not None];return {'n':len(a),'median':float(np.median(a)) if a else None,'q1':float(np.percentile(a,25)) if a else None,'q3':float(np.percentile(a,75)) if a else None}

def mean(values):
 a=[v for v in values if v is not None];return float(np.mean(a)) if a else None

def checked(e,m,x,ids,truth):
 assert e['train_ids_before']==ids and len(e['raw_bits'])==100 and model_hash(m)==e['model_sha256'];qd,b=fm_to_qubo(m);q=matrix(qd,23);pred=np.einsum('bi,ij,bj->b',x,q,x)+b;np.testing.assert_allclose(pred,e['fm_predictions'],atol=1e-10,rtol=0)
 with torch.no_grad():assert np.max(abs(pred-m(torch.tensor(x,dtype=torch.float32)).numpy()))<1e-4
 bits=np.array(e['raw_bits']);assert bits.shape==(100,23) and np.isin(bits,[0,1]).all();i,f=select(bits,pred,x,set(ids));assert i==e['selected_id']
 for k,v in f.items():assert e[k]==v
 h,j=qubo_to_ising_coefficients(q);s=max([abs(v) for v in h.values()]+[abs(v) for v in j.values()]+[0.]);assert s==e['base_Ising_scale_S'] and e['lambda']==s*e['alpha']
 if i is not None:
  assert e['selected_value']==truth[i];gap=float(pred[i]-min(pred[z] for z in range(192) if z not in ids));assert abs(gap-e['selected_unseen_fm_gap'])<1e-10
 else:assert e['selected_unseen_fm_gap'] is None

def analyze():
 v=json.loads((OUT/'json/prevalidation.json').read_text());assert v['source_sha256']=={str(c.relative_to(REPO)):sha(c) for c in CODE};bb,cats,x=problem();truth=np.array([bb.evaluate(c) for c in cats]);opt=truth.min();worst=truth.max();ph=sha(OUT/'json/protocol.json');fixed=[];bbo=[];count=0
 for seed in P['seeds']:
  d=json.loads((OUT/f'json/fixed_seed{seed}.json').read_text());assert d['status']=='completed' and d['protocol_sha256']==ph and len(d['events'])==140 and d['initial_ids']==initial(seed);assert sha(OUT/d['checkpoint'])==d['checkpoint_sha256'];m=TorchFM(23,1);m.load_state_dict(torch.load(OUT/d['checkpoint'],weights_only=True));assert d['initial_values']==truth[d['initial_ids']].tolist()
  for schedule in ['auto','fixed_alpha1_beta']:
   for alpha in P['alphas']:
    es=[e for e in d['events'] if e['schedule']==schedule and e['alpha']==alpha];assert len(es)==10 and sorted(e['rep'] for e in es)==list(range(10))
    for e in es:
     checked(e,m,x,d['initial_ids'],truth)
     if schedule!='auto':np.testing.assert_array_equal(e['neal_info']['beta_range'],d['reference_beta_range'])
    fixed.append({'seed':seed,'alpha':alpha,'schedule':schedule,'feasible_rate':mean([e['raw_feasible_rate'] for e in es]),'candidate_rate':mean([e['selected_id'] is not None for e in es]),'selected_gap':mean([e['selected_unseen_fm_gap'] for e in es]),'gap_defined_batches':sum(e['selected_id'] is not None for e in es),'min_capture_rate':mean([e['selected_unseen_fm_gap'] is not None and e['selected_unseen_fm_gap']<1e-8 for e in es]),'selected_true':mean([e['selected_value'] for e in es]),'initial_plus_one_regret':mean([(min(min(d['initial_values']),e['selected_value'] if e['selected_value'] is not None else float('inf'))-opt)/(worst-opt) for e in es]),'unique_unseen':mean([e['unique_unseen_count'] for e in es]),'beta_range':es[0]['neal_info']['beta_range'],'beta_final_lambda':es[0]['neal_info']['beta_range'][1]*es[0]['lambda']})
  for alpha in P['alphas']:
   d=json.loads((OUT/f'json/bbo_alpha{alpha}_seed{seed}.json').read_text());assert d['status']=='completed' and d['protocol_sha256']==ph and len(d['events'])==70 and d['initial_ids']==initial(seed);assert sha(OUT/d['checkpoint'])==d['checkpoint_sha256'];states=torch.load(OUT/d['checkpoint'],weights_only=False);assert len(states)==70;ids=initial(seed);values=truth[ids].tolist();assert d['initial_values']==values;hit=None;best=[]
   if alpha==100:previous=json.loads((ROOT/f'initial_data_sensitivity/json/LargePenalty-FMQA_initial10_seed{seed}.json').read_text())
   for cycle,(e,cp) in enumerate(zip(d['events'],states),1):
    assert e['alpha']==alpha; m=TorchFM(23,1);m.load_state_dict(cp['state_dict']);assert cp['model_sha256']==e['model_sha256'] and cp['cycle']==e['cycle']==cycle and e['model_seed']==seed+100000*(cycle-1) and e['sampler_seed']==seed*100000+cycle*1000 and e['evaluations_before']==len(ids);checked(e,m,x,ids,truth);count+=1
    if alpha==100:
     for key in ['raw_bits','model_sha256','train_ids_before','fm_predictions','selected_id','selected_value','best_so_far','evaluations_before','evaluations_after']:assert e[key]==previous['events'][cycle-1][key],key
    if e['selected_id'] is not None:ids.append(e['selected_id']);values.append(e['selected_value'])
    assert len(ids)==len(set(ids))<=80 and e['evaluations_after']==len(ids) and e['best_so_far']==min(values);best.append(min(values))
    if hit is None and min(values)==opt:hit=len(ids)
   assert d['final_ids']==ids and d['final_values']==values and d['actual_evaluations']==len(ids) and d['final_best']==min(values)
   bbo.append({'seed':seed,'alpha':alpha,'final_best':min(values),'final_regret':float((min(values)-opt)/(worst-opt)),'success':bool(min(values)==opt),'firsthit_score81':hit if hit is not None else 81,'candidate_none_fraction':mean([e['selected_id'] is None for e in d['events']]),'feasible_rate':mean([e['raw_feasible_rate'] for e in d['events']]),'median_selected_gap':trip([e['selected_unseen_fm_gap'] for e in d['events']])['median'],'actual_evaluations':len(ids),'best_by_cycle':[min(d['initial_values'])]+best})
 groups=[]
 for alpha in P['alphas']:
  rows=[r for r in bbo if r['alpha']==alpha];g={'alpha':alpha,'bbo':{k:trip([r[k] for r in rows]) for k in ['final_best','final_regret','firsthit_score81','candidate_none_fraction','feasible_rate','median_selected_gap','actual_evaluations']},'successes':sum(r['success'] for r in rows),'fixed':{}}
  for schedule in ['auto','fixed_alpha1_beta']:
   rs=[r for r in fixed if r['alpha']==alpha and r['schedule']==schedule];g['fixed'][schedule]={k:trip([r[k] for r in rs]) for k in ['feasible_rate','candidate_rate','selected_gap','selected_true','min_capture_rate','unique_unseen','beta_final_lambda','initial_plus_one_regret']}
  groups.append(g)
 tests=[]
 for alpha in P['alphas']:
  if alpha==100:continue
  for scope,metric in [('bbo','final_regret'),('bbo','firsthit_score81'),('bbo','candidate_none_fraction'),('fixed','feasible_rate'),('fixed','selected_gap')]:
   rs=bbo if scope=='bbo' else [r for r in fixed if r['schedule']=='auto'];diff=[];seeds=[]
   for seed in P['seeds']:
    a=next(r for r in rs if r['seed']==seed and r['alpha']==alpha)[metric];b=next(r for r in rs if r['seed']==seed and r['alpha']==100)[metric]
    if a is not None and b is not None:diff.append(a-b);seeds.append(seed)
   pv=float(wilcoxon(diff,alternative='two-sided',zero_method='wilcox',method='auto').pvalue) if any(abs(z)>1e-12 for z in diff) else 1.
   tests.append({'alpha':alpha,'reference':100,'scope':scope,'metric':metric,'seeds':seeds,'paired_differences':diff,'p_raw':pv,'p_bonferroni':min(1.,30*pv)})
 assert len(tests)==30 and count==2450
 summary={'status':'completed','runs':35,'BBO_cycles':count,'fixed_batches':700,'true_opt':float(opt),'true_worst':float(worst),'groups':groups,'paired_tests':tests,'fixed_per_seed':fixed,'bbo_per_seed':bbo};save(OUT/'json/summary.json',summary);save(OUT/'json/artifact_verification.json',{'status':'passed','BBO_models':2450,'fixed_models':5,'fixed_batches':700,'previous_alpha100_complete_reproductions':5,'failed_or_dropped_runs':0,'checks':'protocol/source/checkpoint/model hashes,FM/QUBO parity,raw candidate filters,alpha/S/lambda,truth/initial/budgets,all70cycles reference parity'});report(summary);plot(summary);print('analysis passed',flush=True)

def fmt(t):
 return '未定義' if t['median'] is None else f"{t['median']:.5g} [{t['q1']:.5g}, {t['q3']:.5g}]"

def report(s):
 lines=['# 大ペナルティ感度の結果','','初期10件、rank1 AdamW wd0.01、lr0.1、120 epochs、100reads、1000sweeps。α100が既定LargePenalty-FMQA。他は固定αの感度variant。5Seedすべて初期最適解なし。BB上限80、最大70サイクル。35Runすべて完了。全値は5Seed中央値 [Q1,Q3]。','','## 再学習BBO（自動beta）','','| α | 最終best eV | 最終Regret | 成功数 | 初到達score | 候補なし率 | Raw充足率 | 実評価数 |','|---:|---:|---:|---:|---:|---:|---:|---:|']
 for g in s['groups']:
  b=g['bbo'];lines.append(f"| {g['alpha']} | {fmt(b['final_best'])} | {fmt(b['final_regret'])} | {g['successes']}/5 | {fmt(b['firsthit_score81'])} | {fmt(b['candidate_none_fraction'])} | {fmt(b['feasible_rate'])} | {fmt(b['actual_evaluations'])} |")
 lines+=['','初到達score81は未到達を示す打切りコードで、81回評価した意味ではない。上限打切りを収束とはしない。','','## 固定FM（初期モデル、再学習なし）','','10独立反復をモデル内平均し、5モデル間で中央値/IQR。条件付きgap・真値は候補がある反復のみ、欠損は0にせず定義数をJSONに保存。1反復の論理予算は初期10＋最大1。反復は連続BBOではない。','','| 日程 | α | Raw充足率 | 候補獲得率 | 選択FM gap eV | 選択真値 eV | FM最小捕捉率 | β_final λ |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for schedule in ['auto','fixed_alpha1_beta']:
  for g in s['groups']:
   a=g['fixed'][schedule];lines.append(f"| {schedule} | {g['alpha']} | {fmt(a['feasible_rate'])} | {fmt(a['candidate_rate'])} | {fmt(a['selected_gap'])} | {fmt(a['selected_true'])} | {fmt(a['min_capture_rate'])} | {fmt(a['beta_final_lambda'])} |")
 lines+=['','## Seed別BBO','','| Seed | α | best eV | 到達score | 実評価数 |','|---:|---:|---:|---:|---:|']
 for r in s['bbo_per_seed']:lines.append(f"| {r['seed']} | {r['alpha']} | {r['final_best']} | {r['firsthit_score81']} | {r['actual_evaluations']} |")
 lines+=['','## 10cycle時点の全Seed（記述的な途中経過）','','実測best_by_cycleの10cycleを事後に抜き出した例。追加の仮説検定はしない。初期10＋最大10件で、候補なしのalpha1では実評価数が20未満の場合がある。alpha3以上は全Runでこの時点が20評価。全Seedを示し、悪化例のみを選ばない。','','| Seed | '+ ' | '.join(str(a) for a in P['alphas'])+' |','|---:|'+ '---:|'*len(P['alphas'])]
 for seed in P['seeds']:
  rs={r['alpha']:r for r in s['bbo_per_seed'] if r['seed']==seed};lines.append('| '+str(seed)+' | '+' | '.join(f"{rs[a]['best_by_cycle'][10]:.4f}" for a in P['alphas'])+' |')
 lines+=['','## ペア検定','','α100基準、固定自動beta2指標＋BBO3指標、6比較×5指標＝30検定。Wilcoxon両側、Bonferroni30。5Seed単位。非有意を同等性と解釈しない。','','| α | 範囲 | 指標 | n | 差の中央値 | 補正p |','|---:|---|---|---:|---:|---:|']
 for t in s['paired_tests']:lines.append(f"| {t['alpha']} | {t['scope']} | {t['metric']} | {len(t['seeds'])} | {np.median(t['paired_differences']):.6g} | {t['p_bonferroni']:.6g} |")
 (OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')

def plot(s):
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});a=np.array(P['alphas']);gs=s['groups'];files=[]
 specs=[('fixed_feasibility','Raw feasible fraction','fixed','feasible_rate'),('fixed_FM_gap','Selected unseen FM gap (eV)','fixed','selected_gap'),('fixed_candidate_rate','Batch candidate frequency','fixed','candidate_rate'),('BBO_final_regret','Final normalized regret','bbo','final_regret'),('BBO_first_hit','First hit score (81 = no hit)','bbo','firsthit_score81'),('BBO_candidate_none','Cycle candidate-none fraction','bbo','candidate_none_fraction')]
 def draw(ax,spec):
  name,ylabel,scope,key=spec
  for label in (['auto','fixed_alpha1_beta'] if scope=='fixed' else ['auto']):
   ts=[g['fixed'][label][key] if scope=='fixed' else g['bbo'][key] for g in gs];med=np.array([t['median'] if t['median'] is not None else np.nan for t in ts]);lo=np.array([t['q1'] if t['q1'] is not None else np.nan for t in ts]);hi=np.array([t['q3'] if t['q3'] is not None else np.nan for t in ts]);ax.plot(a,med,'s--' if label=='auto' else 'o:',color='#ff7f0e' if label=='auto' else '#9467bd',label='SA auto beta' if label=='auto' else 'SA fixed alpha=1 beta');ax.fill_between(a,lo,hi,color='#ff7f0e' if label=='auto' else '#9467bd',alpha=.14)
  
  if key in ['feasible_rate','candidate_rate','candidate_none_fraction']:ax.set_ylim(0,1.05)
  if key=='final_regret':ax.set_ylim(0,max(.05,max(g['bbo'][key]['q3'] for g in gs)*1.1))
  ax.set_xscale('log');ax.set_xlabel(r'Normalized penalty $\alpha=\lambda/S$');ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
 def export(fig,name):
  for ext in ['pdf','svg','png']:
   p=OUT/ext/f'{name}.{ext}';fig.savefig(p,dpi=300,bbox_inches='tight');files.append({'path':str(p.relative_to(OUT)),'sha256':sha(p)})
 fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
 for ax,spec in zip(axs.flat,specs):draw(ax,spec)
 export(fig,'large_penalty_overview');plt.close(fig)
 for spec in specs:
  fig,ax=plt.subplots(figsize=(6.6,4.5),layout='constrained');draw(ax,spec);export(fig,spec[0]);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
 for alpha in P['alphas']:
  curves=np.array([r['best_by_cycle'] for r in s['bbo_per_seed'] if r['alpha']==alpha]);med=np.median(curves,axis=0);q1,q3=np.percentile(curves,[25,75],axis=0);line=ax.plot(range(71),med,label=f'alpha={alpha:g}',drawstyle='steps-post')[0];ax.fill_between(range(71),q1,q3,color=line.get_color(),alpha=.08,step='post')
 ax.set_xlabel('Optimization cycle (initial = 0)');ax.set_ylabel('Best objective (eV)');ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.2);export(fig,'BBO_best_by_cycle');plt.close(fig);save(OUT/'json/figure_manifest.json',{'summary_sha256':sha(OUT/'json/summary.json'),'files':files})

if __name__=='__main__':analyze()
