"""Independent initial-count sensitivity audit and plots."""
import sys,json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wilcoxon
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent
sys.path.insert(0,str(OUT/'py'))
from initial_experiment import P,prior,select,PerovskitesEvaluator,save,sha,model_hash
from fm import TorchFM
from fm_to_qubo import fm_to_qubo
from run_bbo import matrix,qubo_to_ising_coefficients
sys.path.insert(0,str(ROOT/'py'))
from validate_xy_ring import reference_evolution

def stat(a):return {k:float(v) for k,v in zip(['median','q1','q3'],np.quantile(a,[.5,.25,.75]))}

def main():
 bb=PerovskitesEvaluator();cats=list(bb.candidates());x=np.array([bb.encode(c) for c in cats]);y=np.array([bb.evaluate(c) for c in cats]);opt=float(y.min());worst=float(y.max());order=np.argsort(x.astype(np.uint64)@(1<<np.arange(23,dtype=np.uint64)));rows=[];qualities=[];torch.set_num_threads(1)
 for method in P['methods']:
  for n in P['initial_counts']:
   for seed in P['seeds']:
    path=OUT/f'json/{method}_initial{n}_seed{seed}.json';d=json.loads(path.read_text());assert d['status']=='completed' and len(d['events'])==80-n and d['protocol_sha256']==sha(OUT/'json/protocol.json')
    base=json.loads((ROOT/f'fixed_fm_pilot/json/seed_{seed}.json').read_text());assert d['initial_ids']==base['initial_dataset']['candidate_ids'][:n] and d['source20_initial_sha256']==base['initial_sha256'];assert sha(OUT/d['checkpoint'])==d['checkpoint_sha256'];states=torch.load(OUT/d['checkpoint'],weights_only=False);assert len(states)==80-n
    ids=list(d['initial_ids']);vals=list(d['initial_values']);np.testing.assert_array_equal(vals,y[ids]);alpha=1.;old=json.loads(prior(method,seed).read_text()) if n==20 else None
    for e,cp in zip(d['events'],states):
     assert e['cycle']==cp['cycle'] and e['train_ids_before']==ids and e['evaluations_before']==len(ids) and len(e['raw_bits'])==100;m=TorchFM(23,1);m.load_state_dict(cp['state_dict']);assert model_hash(m)==e['model_sha256']==cp['model_sha256'];qd,b=fm_to_qubo(m);q=matrix(qd,23);pred=np.array(e['fm_predictions']);np.testing.assert_allclose(pred,np.einsum('bi,ij,bj->b',x,q,x)+b,atol=1e-10,rtol=0)
     with torch.no_grad():np.testing.assert_allclose(pred,m(torch.tensor(x,dtype=torch.float32)).numpy(),atol=1e-4,rtol=0)
     h,j=qubo_to_ising_coefficients(q);s=max([abs(v) for v in h.values()]+[abs(v) for v in j.values()]+[0.]);assert abs(s-e['base_Ising_scale_S'])<1e-10 and abs(e['lambda']-s*e['alpha'])<1e-10
     if method=='XY-FMQAOA':
      assert e['alpha']==e['lambda']==0;div=s if s>0 else 1.;assert e['normalization_divisor']==div;expected=[]
      for angle in e['angle_search']:
       state,_,_=reference_evolution(q/div,[16,3,4],angle['gamma'],angle['beta']);prob=np.abs(state)**2;v=float(prob@(pred/div));assert abs(v-angle['expected_normalized_FM'])<1e-10;expected.append((v,[angle['gamma'],angle['beta']],prob))
      best=min(expected,key=lambda z:z[0]);assert best[1]==e['chosen_angles'];prob=np.array(e['feasible_probabilities']);np.testing.assert_allclose(prob,best[2],atol=1e-10,rtol=0);assert abs(1-prob.sum())<1e-10;cdf=np.cumsum(prob[order]);u=np.random.default_rng(e['sampler_seed']).random(100)*cdf[-1];draws=order[np.searchsorted(cdf,u,side='left')];np.testing.assert_array_equal(draws,e['raw_candidate_ids']);np.testing.assert_array_equal(x[draws],e['raw_bits'])
     elif method=='Adaptive-FMQA':
      assert abs(e['alpha']-alpha)<1e-12;r=e['raw_feasible_rate'];alpha=min(100.,alpha*2) if r<.7 else min(100.,alpha*1.25) if r<1 else max(.01,alpha*.85);assert abs(e['next_alpha']-alpha)<1e-12
     else:assert e['alpha']==e['next_alpha']==100.
     chosen,meta=select(np.array(e['raw_bits']),pred,x,set(ids));assert chosen==e['selected_id'] and all(e[k]==meta[k] for k in meta);unseen=[i for i in range(192) if i not in ids];assert e['unseen_fm_min']==float(pred[unseen].min())
     if e['cycle']==1:
      common=[i for i in range(192) if i not in base['initial_dataset']['candidate_ids']];qualities.append({'seed':seed,'initial_count':n,'method':method,'common172_test_rmse':float(np.sqrt(np.mean((pred[common]-y[common])**2))),'organic_coverage':int(np.sum(x[ids,:16].sum(axis=0)>0)),'train_rmse':float(np.sqrt(np.mean((pred[ids]-y[ids])**2)))})
     if chosen is not None:
      assert chosen not in ids and e['selected_value']==y[chosen] and abs(e['selected_unseen_fm_gap']-(pred[chosen]-pred[unseen].min()))<1e-10;ids.append(chosen);vals.append(e['selected_value'])
     else:assert e['selected_value'] is None and e['selected_unseen_fm_gap'] is None
     assert e['evaluations_after']==len(ids)==len(set(ids))<=80 and e['best_so_far']==min(vals)
     if old:
      prev=old['events'][e['cycle']-1]
      for key in ['model_sha256','fm_predictions','raw_bits','selected_id','selected_value','train_ids_before','evaluations_after','best_so_far']:assert e[key]==prev[key]
    assert ids==d['final_ids'] and vals==d['final_values'] and d['actual_evaluations']==len(ids) and d['final_best']==min(vals)
    initial=min(d['initial_values'])==opt;hit=n if initial else next((e['evaluations_after'] for e in d['events'] if e['best_so_far']==opt),None);gaps=[e['selected_unseen_fm_gap'] for e in d['events'] if e['selected_id'] is not None];none=sum(e['selected_id'] is None for e in d['events'])
    d['metrics']={'final_best':d['final_best'],'final_regret':(d['final_best']-opt)/(worst-opt),'actual_evaluations':d['actual_evaluations'],'first_hit_score81':hit if hit is not None else 81,'candidate_none_fraction':none/(80-n),'candidate_none_cycles':none,'median_unseenFMgap':float(np.median(gaps)) if gaps else None};d['initial_optimum']=initial;d['first_hit']=hit;d['new_optimum']=not initial and hit is not None;rows.append(d)
 groups=[]
 for method in P['methods']:
  for n in P['initial_counts']:
   ds=[d for d in rows if d['method']==method and d['initial_count']==n];g={'method':method,'initial_count':n,'cycles':80-n,'statistics':{},'new_optimum_runs':sum(d['new_optimum'] for d in ds),'new_optimum_denominator':sum(not d['initial_optimum'] for d in ds),'final_optimum_runs':sum(d['final_best']==opt for d in ds),'per_seed':[{'seed':d['seed'],'initial_best':min(d['initial_values']),'initial_optimum':d['initial_optimum'],'first_hit':d['first_hit'],**d['metrics']} for d in ds]}
   for key in ds[0]['metrics']:
    used=[d for d in ds if d['seed']!=101] if key=='first_hit_score81' else ds;values=[d['metrics'][key] for d in used if d['metrics'][key] is not None];g['statistics'][key]={**stat(values),'n':len(values)} if values else {'median':None,'q1':None,'q3':None,'n':0}
   groups.append(g)
 tests=[]
 for method in P['methods']:
  for n in [5,10]:
   for key in ['final_regret','first_hit_score81','candidate_none_fraction','median_unseenFMgap']:
    diffs=[];seeds=[]
    for seed in P['seeds']:
     if key=='first_hit_score81' and seed==101:continue
     a=next(d for d in rows if d['method']==method and d['initial_count']==n and d['seed']==seed);b=next(d for d in rows if d['method']==method and d['initial_count']==20 and d['seed']==seed)
     if a['metrics'][key] is None or b['metrics'][key] is None:continue
     diffs.append(a['metrics'][key]-b['metrics'][key]);seeds.append(seed)
    p=float(wilcoxon(diffs).pvalue) if np.any(diffs) else 1.;tests.append({'method':method,'contrast':f'{n} minus20','metric':key,'seeds':seeds,'paired_differences':diffs,'median_difference':float(np.median(diffs)) if diffs else None,'p_raw':p,'p_bonferroni':min(1.,p*24)})
 save(OUT/'json/initial_FM_quality.json',{'scope':'offline audit on common172 not in original20,no training/ranking use; first FM identical across methods','rows':qualities})
 summary={'status':'completed','runs':45,'cycles':3075,'XY_angle_checks':9225,'raw_outputs':307500,'groups':groups,'paired_tests':tests};save(OUT/'json/summary.json',summary);save(OUT/'json/artifact_verification.json',{'status':'passed','models':3075,'runs':45,'full_previous20_run_reproductions':15,'independent_XY_angle_checks':9225,'checks':'initial nesting and values,source/protocol/checkpoint/model hashes,FM/QUBO,raw sampling/selection/gaps/alpha/budget,all20controltrajectories unchanged; no failed/dropped runs'});report(summary,qualities);plot(rows,groups,qualities);print(json.dumps(summary,indent=2))

def report(s,qualities):
 lines=['# 初期5／10／20件の比較','','同じ100出力・rank1 AdamW wd0.01、lr0.1、120epochs。BB上限80を共通化し、初期5/10/20に対し最大75/70/60サイクル。45Runを新規計算。中央値 [Q1,Q3]、初回到達は共通の初期最適解なし4Seed、他は5Seed。','','| 手法 | 初期件数 | 新規最適解発見 | 初回到達評価数／未到達81 | 最終best eV | 実評価数 | 候補なし割合 |','|---|---:|---:|---:|---:|---:|---:|']
 for g in s['groups']:
  fmt=lambda k:'{median:.4g} [{q1:.4g},{q3:.4g}]'.format(**g['statistics'][k]);lines.append('| '+g['method']+f" | {g['initial_count']} | {g['new_optimum_runs']}/{g['new_optimum_denominator']} | "+' | '.join(fmt(k) for k in ['first_hit_score81','final_best','actual_evaluations','candidate_none_fraction'])+' |')
 lines+=['','初回到達は初期件数込み。Seed101は20件では初期最適解あり、5/10件では初期になし。到達回数の主比較は共通4Seed42/2024/7/19で揃え、101は下表に残す。未到達81は失敗コードで実評価回数ではない。成功Runだけの中央値を出さない。','','| 手法 | 初期 | Seed | 初期best | 初期最適解 | 初回到達 | 最終best | 実評価数 |','|---|---:|---:|---:|---|---:|---:|---:|']
 for g in s['groups']:
  for r in g['per_seed']:lines.append(f"| {g['method']} | {g['initial_count']} | {r['seed']} | {r['initial_best']:.4f} | {r['initial_optimum']} | {r['first_hit']} | {r['final_best']:.4f} | {r['actual_evaluations']} |")
 lines+=['','## 初期モデルの共通172候補診断','','元20件の全てを除いた172候補で評価し、test集合を初期件数間で共通化。真値は事後監査のみ。最初のモデルは3手法で同じなので重複集計しない。','','| 初期 | 有機カテゴリ網羅数（16中） | Test RMSE eV |','|---:|---:|---:|']
 for n in P['initial_counts']:
  qs=[r for r in qualities if r['initial_count']==n and r['method']=='XY-FMQAOA'];fmt=lambda key:'{median:.4g} [{q1:.4g},{q3:.4g}]'.format(**stat([r[key] for r in qs]));lines.append(f"| {n} | {fmt('organic_coverage')} | {fmt('common172_test_rmse')} |")
 lines+=['','## 事前固定ペア比較','','| 手法 | 比較 | 指標 | ペア差中央値 | raw p | Bonferroni24 p |','|---|---|---|---:|---:|---:|']
 for t in s['paired_tests']:lines.append(f"| {t['method']} | {t['contrast']} | {t['metric']} | {t['median_difference']:.5g} | {t['p_raw']:.5g} | {t['p_bonferroni']:.5g} |")
 lines+=['','総BB予算を揃えた初期配分比較で、同じサイクル数の比較ではない。少ない初期件数には追加探索サイクルを割り当てる。全条件で初期集合のカテゴリ分布も変わり、学習件数だけの純粋な因果効果ではない。既評価/invalidは新規BB評価なし、候補なしno refill。XYは9理想角度点・p1・λ0、100は候補生成shotsで角度期待値推定ではない。']
 (OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')

def plot(rows,groups,quality):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 methods=[('Adaptive-FMQA','#1f77b4','o','-'),('LargePenalty-FMQA','#ff7f0e','s','--'),('XY-FMQAOA','#2ca02c','D','-')]
 def panel(ax,key):
  if key=='best_evaluations':
   for method,color,marker,_ in methods:
    for n,ls in [(5,':'),(10,'--'),(20,'-')]:
     ds=[d for d in rows if d['method']==method and d['initial_count']==n];xx=np.arange(n,min(d['actual_evaluations'] for d in ds)+1);a=np.array([([min(d['initial_values'])]+[e['best_so_far'] for e in d['events'] if e['selected_id'] is not None])[:len(xx)] for d in ds]);q=np.quantile(a,[.25,.5,.75],axis=0);ax.plot(xx,q[1],color=color,ls=ls,marker=marker,markevery=max(1,len(xx)//8),markersize=3,label=f'{method}, init{n}');ax.fill_between(xx,q[0],q[2],color=color,alpha=.05)
   ax.set_xlabel('Actual BB evaluations');ax.set_ylabel('Best-so-far bandgap (eV)');ax.axhline(1.5249,color='gray',ls='-.',lw=.8)
  elif key=='initial_test_RMSE':
   gs=[stat([q['common172_test_rmse'] for q in quality if q['initial_count']==n and q['method']=='XY-FMQAOA']) for n in P['initial_counts']];v=lambda a:np.array([g[a] for g in gs]);ax.errorbar(P['initial_counts'],v('median'),yerr=[v('median')-v('q1'),v('q3')-v('median')],marker='o',capsize=4,color='#555555',label='Initial FM, median / IQR');ax.set_xlabel('Initial evaluations');ax.set_ylabel('Common172 test RMSE (eV)');ax.set_xticks(P['initial_counts'])
  else:
   metric={'first_hit':'first_hit_score81','final_best':'final_best','candidate_fraction':'candidate_none_fraction','FM_gap':'median_unseenFMgap'}[key]
   for method,color,marker,ls in methods:
    gs=[next(g for g in groups if g['method']==method and g['initial_count']==n)['statistics'][metric] for n in P['initial_counts']];v=lambda a:np.array([g[a] for g in gs]);ax.errorbar(P['initial_counts'],v('median'),yerr=[v('median')-v('q1'),v('q3')-v('median')],color=color,marker=marker,ls=ls,capsize=4,label=method)
   ax.set_xticks(P['initial_counts']);ax.set_xlabel('Initial evaluations');ax.set_ylabel({'first_hit':'Evaluations to optimum (81: not reached)','final_best':'Final best bandgap (eV)','candidate_fraction':'Fraction of cycles without a candidate','FM_gap':'Per-run median unseen FM gap (eV)'}[key])
  ax.grid(alpha=.2);ax.legend(fontsize=6 if key=='best_evaluations' else 8)
 keys=['best_evaluations','first_hit','final_best','candidate_fraction','FM_gap','initial_test_RMSE'];fig,axs=plt.subplots(2,3,figsize=(16,9),layout='constrained')
 for ax,key in zip(axs.flat,keys):panel(ax,key)
 for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{ext}/initial_count_overview.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
 for key in keys:
  fig,ax=plt.subplots(figsize=(7,4.8),layout='constrained');panel(ax,key)
  for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{ext}/{key}.{ext}',dpi=300,bbox_inches='tight')
  plt.close(fig)
 save(OUT/'json/figure_manifest.json',{'summary_sha256':sha(OUT/'json/summary.json'),'files':[{'path':str(p.relative_to(OUT)),'sha256':sha(p)} for ext in ['pdf','svg','png'] for p in sorted((OUT/ext).glob('*.'+ext))]})

if __name__=='__main__':main()
