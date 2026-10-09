"""Independent audits, paired count statistics and scientific static figures."""
import sys,json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wilcoxon
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent
sys.path.insert(0,str(OUT/'py'))
from experiment import P,CFG,baseline,select,PerovskitesEvaluator,save,sha,model_hash,matrix,qubo_to_ising_coefficients
from fm import TorchFM
from fm_to_qubo import fm_to_qubo
sys.path.insert(0,str(ROOT/'py'))
from validate_xy_ring import reference_evolution

def stat(a):return {k:float(v) for k,v in zip(['median','q1','q3'],np.quantile(a,[.5,.25,.75]))}

def main():
 bb=PerovskitesEvaluator();cats=list(bb.candidates());x=np.array([bb.encode(c) for c in cats]);y=np.array([bb.evaluate(c) for c in cats]);opt=float(y.min());worst=float(y.max());order=np.argsort(x.astype(np.uint64)@(1<<np.arange(23,dtype=np.uint64)));rows=[];torch.set_num_threads(1)
 for method in P['methods']:
  for count in P['counts']:
   for seed in P['seeds']:
    path=OUT/f'json/{method}_n{count}_seed{seed}.json';d=json.loads(path.read_text());assert d['status']=='completed' and len(d['events'])==60 and d['protocol_sha256']==sha(OUT/'json/protocol.json')
    original=json.loads((ROOT/f'fixed_fm_pilot/json/seed_{seed}.json').read_text());assert d['initial_ids']==original['initial_dataset']['candidate_ids'] and d['initial_sha256']==original['initial_sha256'];assert sha(OUT/d['checkpoint'])==d['checkpoint_sha256']
    states=torch.load(OUT/d['checkpoint'],weights_only=False);assert len(states)==60;ids=list(d['initial_ids']);vals=list(d['initial_values']);np.testing.assert_array_equal(vals,y[ids]);alpha=1.;prior=json.loads(baseline(method,seed).read_text()) if count==10 else None
    for e,cp in zip(d['events'],states):
     assert e['cycle']==cp['cycle'] and e['train_ids_before']==ids and e['evaluations_before']==len(ids) and len(e['raw_bits'])==count
     m=TorchFM(23,1);m.load_state_dict(cp['state_dict']);assert model_hash(m)==e['model_sha256']==cp['model_sha256'];qd,b=fm_to_qubo(m);q=matrix(qd,23);pred=np.array(e['fm_predictions']);np.testing.assert_allclose(pred,np.einsum('bi,ij,bj->b',x,q,x)+b,atol=1e-10,rtol=0)
     with torch.no_grad():np.testing.assert_allclose(pred,m(torch.tensor(x,dtype=torch.float32)).numpy(),atol=1e-4,rtol=0)
     h,j=qubo_to_ising_coefficients(q);s=max([abs(v) for v in h.values()]+[abs(v) for v in j.values()]+[0.]);assert abs(s-e['base_Ising_scale_S'])<1e-10 and abs(e['lambda']-s*e['alpha'])<1e-10
     if method=='XY-FMQAOA':
      assert e['alpha']==e['lambda']==0;div=s if s>0 else 1.;assert e['normalization_divisor']==div;expected=[]
      for angle in e['angle_search']:
       state,_,_=reference_evolution(q/div,[16,3,4],angle['gamma'],angle['beta']);p=np.abs(state)**2;v=float(p@(pred/div));assert abs(v-angle['expected_normalized_FM'])<1e-10;expected.append((v,[angle['gamma'],angle['beta']],p))
      best=min(expected,key=lambda z:z[0]);assert best[1]==e['chosen_angles'];prob=np.array(e['feasible_probabilities']);np.testing.assert_allclose(prob,best[2],atol=1e-10,rtol=0);assert abs(1-prob.sum())<1e-10;cdf=np.cumsum(prob[order]);u=np.random.default_rng(e['sampler_seed']).random(count)*cdf[-1];draws=order[np.searchsorted(cdf,u,side='left')];np.testing.assert_array_equal(draws,e['raw_candidate_ids']);np.testing.assert_array_equal(x[draws],e['raw_bits'])
     elif method=='Adaptive-FMQA':
      assert abs(e['alpha']-alpha)<1e-12;r=e['raw_feasible_rate'];alpha=min(100.,alpha*2) if r<.7 else min(100.,alpha*1.25) if r<1 else max(.01,alpha*.85);assert abs(e['next_alpha']-alpha)<1e-12
     else:assert e['alpha']==e['next_alpha']==100.
     chosen,meta=select(np.array(e['raw_bits']),pred,x,set(ids));assert chosen==e['selected_id'] and all(e[k]==meta[k] for k in meta);unseen=[i for i in range(192) if i not in ids];assert e['unseen_fm_min']==float(pred[unseen].min()) and e['global_feasible_fm_min']==float(pred.min())
     if chosen is not None:
      assert chosen not in ids and e['selected_value']==y[chosen];assert abs(e['selected_unseen_fm_gap']-(pred[chosen]-pred[unseen].min()))<1e-10 and abs(e['selected_global_fm_gap']-(pred[chosen]-pred.min()))<1e-10;ids.append(chosen);vals.append(e['selected_value'])
     else:assert e['selected_value'] is None and e['selected_unseen_fm_gap'] is None
     assert e['evaluations_after']==len(ids)==len(set(ids))<=80 and e['best_so_far']==min(vals)
     if prior:
      old=prior['events'][e['cycle']-1]
      for key in ['model_sha256','fm_predictions','raw_bits','selected_id','selected_value','train_ids_before','evaluations_after','best_so_far']:assert e[key]==old[key],(method,seed,e['cycle'],key)
    assert d['final_ids']==ids and d['final_values']==vals and d['actual_evaluations']==len(ids) and d['final_best']==min(vals)
    initial=min(d['initial_values'])==opt;hit=20 if initial else next((e['evaluations_after'] for e in d['events'] if e['best_so_far']==opt),None);gaps=[e['selected_unseen_fm_gap'] for e in d['events'] if e['selected_id'] is not None]
    d['metrics']={'final_best':d['final_best'],'final_regret':(d['final_best']-opt)/(worst-opt),'actual_evaluations':d['actual_evaluations'],'first_hit_score81':hit if hit is not None else 81,'zero_candidate_cycles':sum(e['selected_id'] is None for e in d['events']),'median_unseen_FM_gap':float(np.median(gaps)) if gaps else None,'raw_feasible_rate':float(np.mean([e['raw_feasible_rate'] for e in d['events']]))};d['first_hit']=hit;d['initial_optimum']=initial;d['new_optimum']=not initial and hit is not None;rows.append(d)
 groups=[]
 for method in P['methods']:
  for count in P['counts']:
   ds=[d for d in rows if d['method']==method and d['count']==count];non=[d for d in ds if not d['initial_optimum']];gs={'method':method,'count':count,'statistics':{},'new_optimum_runs':sum(d['new_optimum'] for d in ds),'final_optimum_runs':sum(d['final_best']==opt for d in ds),'per_seed':[{'seed':d['seed'],'initial_optimum':d['initial_optimum'],'first_hit':d['first_hit'],**d['metrics']} for d in ds]}
   for key in ds[0]['metrics']:
    use=non if key=='first_hit_score81' else ds;values=[d['metrics'][key] for d in use if d['metrics'][key] is not None];gs['statistics'][key]={**stat(values),'n':len(values)} if values else {'median':None,'q1':None,'q3':None,'n':0}
   groups.append(gs)
 tests=[]
 for method in P['methods']:
  for key in ['final_regret','first_hit_score81','zero_candidate_cycles','median_unseen_FM_gap']:
   diffs=[];seeds=[]
   for seed in P['seeds']:
    a=next(d for d in rows if d['method']==method and d['count']==100 and d['seed']==seed);b=next(d for d in rows if d['method']==method and d['count']==10 and d['seed']==seed)
    if key=='first_hit_score81' and a['initial_optimum']:continue
    if a['metrics'][key] is None or b['metrics'][key] is None:continue
    diffs.append(a['metrics'][key]-b['metrics'][key]);seeds.append(seed)
   p=float(wilcoxon(diffs).pvalue) if np.any(diffs) else 1.;tests.append({'method':method,'metric':key,'contrast':'100 minus10','seeds':seeds,'paired_differences':diffs,'median_difference':float(np.median(diffs)) if diffs else None,'p_raw':p,'p_bonferroni':min(1.,p*12),'n_pairs':len(diffs)})
 # Separate frozen-model XY shot-count counterfactual: same model/angles/uniforms.
 nested=[]
 for d in rows:
  if d['method']!='XY-FMQAOA' or d['count']!=10:continue
  for e in d['events']:
   prob=np.array(e['feasible_probabilities']);pred=np.array(e['fm_predictions']);cdf=np.cumsum(prob[order]);u=np.random.default_rng(e['sampler_seed']).random(100)*cdf[-1];draws=order[np.searchsorted(cdf,u,side='left')];np.testing.assert_array_equal(draws[:10],e['raw_candidate_ids']);a,_=select(x[draws[:10]],pred,x,set(e['train_ids_before']));b,_=select(x[draws],pred,x,set(e['train_ids_before']));ga=float(pred[a]-e['unseen_fm_min']) if a is not None else None;gb=float(pred[b]-e['unseen_fm_min']) if b is not None else None
   if ga is not None:assert gb is not None and gb<=ga+1e-12
   nested.append({'seed':d['seed'],'cycle':e['cycle'],'chosen10':a,'chosen100':b,'gap10':ga,'gap100':gb})
 nested_summary=[]
 for seed in P['seeds']:
  ns=[r for r in nested if r['seed']==seed];nested_summary.append({'seed':seed,'median_gap10':float(np.median([r['gap10'] for r in ns if r['gap10'] is not None])),'median_gap100':float(np.median([r['gap100'] for r in ns if r['gap100'] is not None])),'strictly_better_cycles':sum(r['gap100']<r['gap10']-1e-12 for r in ns if r['gap10'] is not None),'same_cycles':sum(abs(r['gap100']-r['gap10'])<=1e-12 for r in ns if r['gap10'] is not None),'missing10':sum(r['gap10'] is None for r in ns),'missing100':sum(r['gap100'] is None for r in ns)})
 save(OUT/'json/fixed_FM_XY_nested.json',{'scope':'same10runFM/angles and100 uniform draws; first10 prefix; predicted gaps only,no BB confirmations/no BBO claim','rows':nested,'per_seed':nested_summary})
 summary={'status':'completed','runs':30,'cycles':1800,'raw_outputs':99000,'XY_angle_evaluations':5400,'groups':groups,'paired_tests':tests,'fixed_FM_XY_nested':nested_summary};save(OUT/'json/summary.json',summary);save(OUT/'json/artifact_verification.json',{'status':'passed','runs':30,'models':1800,'XY_independent_angle_checks':5400,'complete_previous10_reproductions':15,'checks':'all checkpoint/model/protocol hashes,forward-QUBO,raw selection/alpha/budget/lookup truth,gaps; 15full10trajectories exactly reproduced; fixedFM XY100contains10 and nonworseninggap; no failed or excluded runs'})
 report(summary);plot(rows,groups,nested_summary);print(json.dumps(summary,indent=2))

def report(s):
 lines=['# 10／100出力の再学習最適化比較','','rank1・AdamW wd0.01・lr0.1・120 epochs。初期20件、60サイクル、最大1件採用、BB評価上限80を共通化。3手法×2出力数×5Seedの30Runを新規計算し、過去10出力の15Runを全60サイクル完全再現した。中央値 [Q1,Q3]、初回到達だけ初期最適解なし4Seed、他は5Seed。','','| 手法 | 出力数 | 新規最適解発見 | 初回到達評価数／未到達81 | 最終best eV | 実評価数 | 候補なしサイクル | 採用候補のFM gap eV |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for g in s['groups']:
  fmt=lambda k:'{median:.4g} [{q1:.4g},{q3:.4g}]'.format(**g['statistics'][k]);lines.append('| '+g['method']+f" | {g['count']} | {g['new_optimum_runs']}/4 | "+' | '.join(fmt(k) for k in ['first_hit_score81','final_best','actual_evaluations','zero_candidate_cycles','median_unseen_FM_gap'])+' |')
 lines+=['','採用FM gap＝選択候補のFM予測−その時点の未評価実行可能192候補内FM最小（既評価を除く）。各Runの採用サイクルにおけるgap中央値を計算し、それを5Seedで要約。候補なしをgap0として扱わない。global feasible FM最小との差も各イベントに保存。FM予測の差で、真の目的値の差ではない。','','初回到達は初期20を含む。Seed101は初期から最適解を含み、新規発見と到達スコアの集計から分けた。未到達を81という失敗コードで保持し、未到達Runを除外した中央値を出さない。81は実際に行った評価ではない。最適値1.5249 eV、最悪6.3242 eVは全192候補表の固定値。','','## Seed別結果','','| 手法 | count | Seed | 初期最適解 | 初回到達（None＝未到達） | 最終best | 実評価数 |','|---|---:|---:|---|---:|---:|---:|']
 for g in s['groups']:
  for r in g['per_seed']:lines.append(f"| {g['method']} | {g['count']} | {r['seed']} | {r['initial_optimum']} | {r['first_hit']} | {r['final_best']:.4f} | {r['actual_evaluations']} |")
 lines+=['','## 同一Seedペア差分（100−10）','','| 手法 | 指標 | n | 差分中央値 | raw p | Bonferroni12 p |','|---|---|---:|---:|---:|---:|']
 for t in s['paired_tests']:lines.append(f"| {t['method']} | {t['metric']} | {t['n_pairs']} | {t['median_difference']:.5g} | {t['p_raw']:.5g} | {t['p_bonferroni']:.5g} |")
 lines+=['','## 同じFM・同じ角度でのXY補助診断','','10出力Runの各300固定モデルで、同じ100個の乱数から先頭10と全100を比較。閉ループ100Runとは別。真値の追加評価なし。','','| Seed | gap10中央値 eV | gap100中央値 eV | 改善サイクル | 同値サイクル |','|---:|---:|---:|---:|---:|']
 for r in s['fixed_FM_XY_nested']:lines.append(f"| {r['seed']} | {r['median_gap10']:.5g} | {r['median_gap100']:.5g} | {r['strictly_better_cycles']} | {r['same_cycles']} |")
 lines+=['','100出力は訓練軌跡とAdaptiveのα更新も変えるため、閉ループ平均gapをサンプラー単独の因果効果と呼ばない。SA100は新しいnum_reads=100で実行し、旧1000poolからの部分抽出ではない。XYの角度探索は9理想期待値のままで、100shotsを角度期待値推定へ使っていない。出力数増加の費用をBB予算一致で打ち消したとは主張しない。']
 (OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')

def plot(rows,groups,nested):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 methods=[('Adaptive-FMQA','#1f77b4','o','-'),('LargePenalty-FMQA','#ff7f0e','s','--'),('XY-FMQAOA','#2ca02c','D','-')]
 def panel(ax,key):
  if key=='fixed_FM_XY':
   gs=[stat([r['median_gap'+str(n)] for r in nested]) for n in [10,100]];v=lambda name:np.array([g[name] for g in gs]);ax.errorbar([0,1],v('median'),yerr=[v('median')-v('q1'),v('q3')-v('median')],color='#2ca02c',marker='D',capsize=4,label='XY fixed FM / angles: median, IQR');ax.set_xticks([0,1],['10','100']);ax.set_xlabel('XY shots (nested draws)');ax.set_ylabel('Per-run median unseen FM gap (eV)')
  elif key in ['candidate_shortfall','FM_gap','first_hit']:
   metric={'candidate_shortfall':'zero_candidate_cycles','FM_gap':'median_unseen_FM_gap','first_hit':'first_hit_score81'}[key]
   for j,(method,c,marker,style) in enumerate(methods):
    gs=[next(g for g in groups if g['method']==method and g['count']==n)['statistics'][metric] for n in [10,100]];v=lambda name:np.array([g[name] for g in gs]);ax.errorbar(np.arange(2)+(j-1)*.04,v('median'),yerr=[v('median')-v('q1'),v('q3')-v('median')],color=c,marker=marker,ls=style,capsize=4,label=method)
   ax.set_xticks([0,1],['10','100']);ax.set_xlabel('SA reads / XY shots');ax.set_ylabel({'candidate_shortfall':'Cycles with no candidate','FM_gap':'Per-run median unseen FM gap (eV)','first_hit':'Evaluations to optimum (81: not reached)'}[key])
  else:
   for method,c,marker,style in methods:
    for count in [10,100]:
     ds=[d for d in rows if d['method']==method and d['count']==count];xx=np.arange(20,min(d['actual_evaluations'] for d in ds)+1) if key=='best_evaluations' else np.arange(61)
     a=np.array([([min(d['initial_values'])]+[e['best_so_far'] for e in d['events'] if e['selected_id'] is not None])[:len(xx)] if key=='best_evaluations' else [min(d['initial_values'])]+[e['best_so_far'] for e in d['events']] for d in ds]);q=np.quantile(a,[.25,.5,.75],axis=0);ax.plot(xx,q[1],color=c,marker=marker,ls=style if count==100 else ':',markevery=max(1,len(xx)//8),markersize=3,label=f'{method}, {count}');ax.fill_between(xx,q[0],q[2],color=c,alpha=.07)
   ax.set_xlabel('Actual BB evaluations' if key=='best_evaluations' else 'BBO cycle');ax.set_ylabel('Best-so-far bandgap (eV)');ax.axhline(1.5249,color='gray',lw=.8,ls='-.')
  ax.grid(alpha=.2);ax.legend(fontsize=7)
 keys=['best_evaluations','best_cycles','first_hit','candidate_shortfall','FM_gap','fixed_FM_XY'];fig,axs=plt.subplots(2,3,figsize=(16,9),layout='constrained')
 for ax,key in zip(axs.flat,keys):panel(ax,key)
 for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{ext}/sample_count_overview.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
 for key in keys:
  fig,ax=plt.subplots(figsize=(7,4.8),layout='constrained');panel(ax,key)
  for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{ext}/{key}.{ext}',dpi=300,bbox_inches='tight')
  plt.close(fig)
 save(OUT/'json/figure_manifest.json',{'summary_sha256':sha(OUT/'json/summary.json'),'files':[{'path':str(p.relative_to(OUT)),'sha256':sha(p)} for ext in ['pdf','svg','png'] for p in sorted((OUT/ext).glob('*.'+ext))]})

if __name__=='__main__':main()
