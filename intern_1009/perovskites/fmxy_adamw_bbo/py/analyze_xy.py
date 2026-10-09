"""Independently audit XY gate probabilities, measured selection and BBO budget."""
import sys,json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wilcoxon
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent
sys.path.insert(0,str(OUT/'py'))
from run_xy import P,CFG,select,PerovskitesEvaluator,save,sha,model_hash,OpenQARPCompactXYQAOA,reference_evolution
from fm import TorchFM
from fm_to_qubo import fm_to_qubo
from run_bbo import matrix,qubo_to_ising_coefficients

def stat(a):return {k:float(v) for k,v in zip(['median','q1','q3'],np.quantile(a,[.5,.25,.75]))}

def main():
 bb=PerovskitesEvaluator();cats=list(bb.candidates());x=np.array([bb.encode(c) for c in cats]);truth=np.array([bb.evaluate(c) for c in cats]);order=np.argsort(x.astype(np.uint64)@(1<<np.arange(23,dtype=np.uint64)));rows=[];torch.set_num_threads(1)
 for seed in P['seeds']:
  d=json.loads((OUT/f'json/seed_{seed}.json').read_text());assert d['status']=='completed' and len(d['events'])==60 and sha(OUT/d['checkpoint'])==d['checkpoint_sha256'];basepath=ROOT/f'fixed_fm_pilot/json/seed_{seed}.json';base=json.loads(basepath.read_text());assert d['source_initial_sha256']==sha(basepath) and d['initial_ids']==base['initial_dataset']['candidate_ids'] and d['initial_sha256']==base['initial_sha256']
  states=torch.load(OUT/d['checkpoint'],weights_only=False);ids=list(d['initial_ids']);vals=list(d['initial_values']);np.testing.assert_array_equal(vals,truth[ids]);assert len(states)==60
  for e,cp in zip(d['events'],states):
   assert e['train_ids_before']==ids and e['evaluations_before']==len(ids) and e['cycle']==cp['cycle'];m=TorchFM(23,1);m.load_state_dict(cp['state_dict']);assert model_hash(m)==cp['model_sha256']==e['model_sha256'];pred=np.array(e['fm_predictions']);qd,b=fm_to_qubo(m);q=matrix(qd,23)
   np.testing.assert_allclose(pred,np.einsum('bi,ij,bj->b',x,q,x)+b,atol=1e-10,rtol=0)
   with torch.no_grad():np.testing.assert_allclose(pred,m(torch.tensor(x,dtype=torch.float32)).numpy(),atol=1e-4,rtol=0)
   h,j=qubo_to_ising_coefficients(q);scale=max([abs(v) for v in h.values()]+[abs(v) for v in j.values()]+[0.]);assert abs(scale-e['base_Ising_scale_S'])<1e-10 and e['lambda']==e['alpha']==0;div=scale if scale>0 else 1.;assert div==e['normalization_divisor'];qn=q/div
   # Independent ordered-edge evolution, not the compact encoded circuit.
   expected=[]
   for angle in e['angle_search']:
    state,_,_=reference_evolution(qn,[16,3,4],angle['gamma'],angle['beta']);prob=np.abs(state)**2;val=float(prob@(pred/div));assert abs(val-angle['expected_normalized_FM'])<1e-10 and abs(prob.sum()-1)<1e-10;expected.append((val,[angle['gamma'],angle['beta']],prob))
   best=min(expected,key=lambda z:z[0]);assert best[1]==e['chosen_angles'];prob=np.array(e['feasible_probabilities']);np.testing.assert_allclose(prob,best[2],atol=1e-10,rtol=0)
   cdf=np.cumsum(prob[order]);u=np.random.default_rng(e['shot_seed']).random(10)*cdf[-1];sampleids=order[np.searchsorted(cdf,u,side='left')];np.testing.assert_array_equal(sampleids,e['raw_candidate_ids']);np.testing.assert_array_equal(x[sampleids],e['raw_bits']);chosen,meta=select(x[sampleids],pred,x,set(ids));assert chosen==e['selected_id'] and all(e[k]==meta[k] for k in meta)
   if chosen is not None:assert chosen not in ids and e['selected_value']==truth[chosen];ids.append(chosen);vals.append(e['selected_value'])
   else:assert e['selected_value'] is None
   assert e['evaluations_after']==len(ids)==len(set(ids))<=80 and e['best_so_far']==min(vals)
  assert ids==d['final_ids'] and vals==d['final_values'] and d['actual_evaluations']==len(ids) and d['final_best']==min(vals)
  # First-cycle model exactly matches proposed SA before trajectories diverge.
  sa=json.loads((ROOT/f'fmqa_adamw_bbo/json/proposed_Adaptive-FMQA_seed{seed}.json').read_text());assert sa['events'][0]['model_sha256']==d['events'][0]['model_sha256']
  d['method']='XY-FMQAOA';rows.append(d)
 for method in ['Adaptive-FMQA','LargePenalty-FMQA']:
  for seed in P['seeds']:d=json.loads((ROOT/f'fmqa_adamw_bbo/json/proposed_{method}_seed{seed}.json').read_text());rows.append(d)
 groups=[]
 for method in ['XY-FMQAOA','Adaptive-FMQA','LargePenalty-FMQA']:
  ds=[d for d in rows if d['method']==method];per=[]
  for d in ds:
   initial=min(d['initial_values'])==truth.min();hit=20 if initial else next((e['evaluations_after'] for e in d['events'] if e['best_so_far']==truth.min()),None)
   per.append({'seed':d['seed'],'initial_best':min(d['initial_values']),'final_best':d['final_best'],'actual_evaluations':d['actual_evaluations'],'initial_optimum':bool(initial),'newly_found_optimum':bool(not initial and d['final_best']==truth.min()),'first_optimum_evaluations':hit,'zero_candidate_cycles':sum(e['selected_id'] is None for e in d['events'])})
  groups.append({'method':method,'statistics':{key:stat([d[key] for d in per]) for key in ['final_best','actual_evaluations','zero_candidate_cycles']},'new_optimum_runs':sum(d['newly_found_optimum'] for d in per),'final_optimum_runs':sum(int(d['final_best']==truth.min()) for d in per),'per_seed':per})
 tests=[]
 for method in ['Adaptive-FMQA','LargePenalty-FMQA']:
  for key in ['final_regret','actual_evaluations']:
   diffs=[]
   for seed in P['seeds']:
    a=next(d for d in rows if d['method']=='XY-FMQAOA' and d['seed']==seed);b=next(d for d in rows if d['method']==method and d['seed']==seed);diffs.append((a['final_best']-b['final_best'])/(truth.max()-truth.min()) if key=='final_regret' else a[key]-b[key])
   p=float(wilcoxon(diffs).pvalue) if np.any(diffs) else 1.;tests.append({'contrast':'XY minus '+method,'metric':key,'paired_differences':diffs,'median_difference':float(np.median(diffs)),'p_raw':p,'p_bonferroni':min(1.,p*4)})
 s={'status':'completed','XY_runs':5,'XY_cycles':300,'ideal_angle_evaluations':2700,'shots':3000,'groups':groups,'paired_tests':tests};save(OUT/'json/summary.json',s);save(OUT/'json/artifact_verification.json',{'status':'passed','models':300,'independent_angle_reference_checks':2700,'checks':'checkpoint/hash/FM-QUBO parity; independent XY ordered-edge probabilities and chosen angles; shot RNG; candidate ranking; truth/train/evaluation budget; same initialdata and first FM as SA; no failures or exclusions'})
 report(s);plot(rows);print(json.dumps(s,indent=2))

def report(s):
 lines=['# FM-XYQAOA再学習最適化','','同じrank1・AdamW・wd0.01、lr0.1、120 epochs。初期20件、60サイクル、10出力/cycle、最大1件採用、評価上限80。XYはp1・λ0・product W・Ring順序付き辺ゲート積、9角度点の理想期待値で毎サイクル選択。5 Seed・300サイクル・2700角度評価・3000shots完了。数値は中央値 [Q1,Q3]。','','| 手法 | 最終best eV | 実評価数 | 候補なしサイクル | 最適解保持 | 新規発見 |','|---|---:|---:|---:|---:|---:|']
 for g in s['groups']:
  fmt=lambda k:'{median:.4g} [{q1:.4g},{q3:.4g}]'.format(**g['statistics'][k]);lines.append('| '+g['method']+' | '+' | '.join(fmt(k) for k in ['final_best','actual_evaluations','zero_candidate_cycles'])+f" | {g['final_optimum_runs']}/5 | {g['new_optimum_runs']}/4 |")
 lines+=['','Seed101は初期20件に最適解があり、新規発見の分母は4。最適値は全192候補表の1.5249 eV、hydrazinium/Sn/I。物理実験の新規最適性を主張しない。','','| 手法 | Seed | 初期best | 最終best | 実評価数 | 最適値初回到達評価数 |','|---|---:|---:|---:|---:|---:|']
 for g in s['groups']:
  for d in g['per_seed']:lines.append(f"| {g['method']} | {d['seed']} | {d['initial_best']:.4f} | {d['final_best']:.4f} | {d['actual_evaluations']} | {d['first_optimum_evaluations']} |")
 lines+=['','## 補正検定','','| 比較 | 指標 | ペア差分中央値 | raw p | Bonferroni p |','|---|---|---:|---:|---:|']
 for t in s['paired_tests']:lines.append(f"| {t['contrast']} | {t['metric']} | {t['median_difference']:.5g} | {t['p_raw']:.5g} | {t['p_bonferroni']:.5g} |")
 lines+=['','BB予算と出力数は揃えたが、SAの1000sweepsとQAOAの9理想期待値評価は同じ計算費用ではない。有限shotsの角度探索、大域的角度最適性、実機ゲート合成、標準X-QAOAは未検証。終了は60サイクルの打ち切りであり、収束判定ではない。']
 (OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')

def plot(rows):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 def panel(ax,key):
  for method,color,marker,style in [('Adaptive-FMQA','#1f77b4','o','-'),('LargePenalty-FMQA','#ff7f0e','s','--'),('XY-FMQAOA','#2ca02c','D','-')]:
   ds=[d for d in rows if d['method']==method]
   if key=='best_evaluations':
    xx=np.arange(20,min(d['actual_evaluations'] for d in ds)+1);a=np.array([([min(d['initial_values'])]+[e['best_so_far'] for e in d['events'] if e['selected_id'] is not None])[:len(xx)] for d in ds])
   else:
    xx=np.arange(61);a=np.array([[min(d['initial_values'])]+[e['best_so_far'] for e in d['events']] if key=='best_cycles' else [20]+[e['evaluations_after'] for e in d['events']] for d in ds])
   q=np.quantile(a,[.25,.5,.75],axis=0);ax.plot(xx,q[1],color=color,marker=marker,linestyle=style,markevery=max(1,len(xx)//8),markersize=4,label=method);ax.fill_between(xx,q[0],q[2],color=color,alpha=.12)
  ax.set_xlabel('Actual BB evaluations' if key=='best_evaluations' else 'BBO cycle');ax.set_ylabel('Cumulative BB evaluations' if key=='evaluations_cycles' else 'Best-so-far bandgap (eV)');ax.grid(alpha=.2);ax.legend(fontsize=8)
  if key.startswith('best'):ax.axhline(1.5249,color='gray',ls='-.',lw=.8)
 keys=['best_evaluations','best_cycles','evaluations_cycles'];fig,axs=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
 for ax,k in zip(axs,keys):panel(ax,k)
 for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{ext}/fmxy_bbo_overview.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
 for k in keys:
  fig,ax=plt.subplots(figsize=(6.8,4.7),layout='constrained');panel(ax,k)
  for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{ext}/{k}.{ext}',dpi=300,bbox_inches='tight')
  plt.close(fig)
 save(OUT/'json/figure_manifest.json',{'files':[{'path':str(p.relative_to(OUT)),'sha256':sha(p)} for ext in ['pdf','svg','png'] for p in sorted((OUT/ext).glob('*.'+ext))]})

if __name__=='__main__':main()
