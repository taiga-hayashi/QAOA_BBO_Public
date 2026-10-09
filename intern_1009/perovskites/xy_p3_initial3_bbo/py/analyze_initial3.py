"""Checkpoint, chosen-state and BB-budget audit; initial3 BBO reports."""
import sys,json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wilcoxon
from run_initial3 import OUT,ROOT,REPO,P,CODE,initial,candidates,problem,sha,save,model_hash,reference_layers
sys.path.insert(0,str(ROOT/'large_penalty_diagnostic/py'))
from analyze_penalty import checked,TorchFM,trip,plt
from run_bbo import fm_to_qubo,matrix

def main():
 torch.set_num_threads(1);v=json.loads((OUT/'json/prevalidation.json').read_text());assert v['status']=='passed' and v['source_sha256']=={str(c.relative_to(REPO)):sha(c) for c in CODE};bb,cats,x=problem();truth=np.array([bb.evaluate(c) for c in cats]);opt=float(truth.min());worst=float(truth.max());order=np.argsort(x.astype(np.uint64)@(1<<np.arange(23,dtype=np.uint64)));rows=[];models=angles=selectedchecks=0;firsthashes={}
 for seed in P['seeds']:
  for method in P['methods']:
   d=json.loads((OUT/f'json/{method}_seed{seed}.json').read_text());assert d['status']=='completed' and d['protocol_sha256']==sha(OUT/'json/protocol.json') and d['seed']==seed and d['method']==method and len(d['events'])==77 and d['initial_ids']==initial(seed);assert sha(OUT/d['checkpoint'])==d['checkpoint_sha256'];states=torch.load(OUT/d['checkpoint'],weights_only=False);assert len(states)==77;ids=initial(seed);vals=truth[ids].tolist();assert d['initial_values']==vals and min(vals)>opt;hit=None;curve=[min(vals)];evalcurve=[(3,min(vals))];gaps=[];none=0;rawrates=[];predrmse=[];accepted=[]
   for cycle,(e,cp) in enumerate(zip(d['events'],states),1):
    assert e['cycle']==cp['cycle']==cycle and e['model_seed']==seed+100000*(cycle-1) and e['evaluations_before']==len(ids) and e['sampler_seed']==seed*100000+cycle*1000;m=TorchFM(23,1);m.load_state_dict(cp['state_dict']);assert cp['model_hash']==e['model_sha256'];checked(e,m,x,ids,truth);models+=1
    if cycle==1:
     old=firsthashes.setdefault(seed,model_hash(m));assert old==model_hash(m)==next(c for c in v['initial3_checks'] if c['seed']==seed)['model_hash']
    pred=np.array(e['fm_predictions']);qd,b=fm_to_qubo(m);q=matrix(qd,23)
    if method.startswith('XY'):
     p=1 if method=='XY-p1' else 3;assert e['p']==p and e['alpha']==e['lambda']==0 and e['raw_feasible_rate']==1.;search=e['angle_search'];assert len(search)==128
     for index,(r,(gs,bs)) in enumerate(zip(search,candidates(p,seed,cycle))):
      assert r['index']==index and r['gammas']==gs and r['betas']==bs and abs(1-r['feasible_mass'])<1e-10 and np.isfinite(r['expected_FM']);angles+=1
     best=min(search,key=lambda z:z['expected_FM']);assert best==e['best_angles'];div=e['base_Ising_scale_S'] if e['base_Ising_scale_S']>0 else 1.;assert e['normalization_divisor']==div;prob=abs(reference_layers(q/div,[16,3,4],best['gammas'],best['betas']))**2;np.testing.assert_allclose(prob,e['feasible_probabilities'],atol=1e-10,rtol=0);assert abs(1-prob.sum())<1e-10 and abs(float(prob@pred)-best['expected_FM'])<1e-10;selectedchecks+=1
     saved=np.array(e['feasible_probabilities']);cdf=np.cumsum(saved[order]);u=np.random.default_rng(e['sampler_seed']).random(100)*cdf[-1];draws=order[np.searchsorted(cdf,u,side='left')];np.testing.assert_array_equal(draws,e['raw_candidate_ids']);np.testing.assert_array_equal(x[draws],e['raw_bits'])
    else:assert e['alpha']==1000 and e['lambda']==1000*e['base_Ising_scale_S'] and e['neal_info']['beta_schedule_type']=='geometric'
    rawrates.append(e['raw_feasible_rate']);none+=e['selected_id'] is None
    if e['selected_id'] is not None:
     i=e['selected_id'];ids.append(i);vals.append(e['selected_value']);gaps.append(e['selected_unseen_fm_gap']);accepted.append(e['selected_value']);evalcurve.append((len(ids),min(vals)))
    assert e['evaluations_after']==len(ids) and len(ids)==len(set(ids))<=80 and e['best_so_far']==min(vals);curve.append(min(vals))
    if hit is None and min(vals)==opt:hit=len(ids)
   assert ids==d['final_ids'] and vals==d['final_values'] and min(vals)==d['final_best'] and len(ids)==d['actual_evaluations'];r={'method':method,'seed':seed,'final_best':min(vals),'final_regret':(min(vals)-opt)/(worst-opt),'success':bool(min(vals)==opt),'firsthit_score81':hit if hit is not None else 81,'actual_evaluations':len(ids),'candidate_none_fraction':none/77,'median_FM_gap':trip(gaps)['median'],'raw_feasible_fraction':float(np.mean(rawrates)),'generation_seconds':sum(e['generation_seconds'] for e in d['events']),'training_seconds':sum(e['training_seconds'] for e in d['events']),'best_by_cycle':curve,'best_by_evaluation':evalcurve};rows.append(r)
 assert models==1155 and angles==98560 and selectedchecks==770;groups=[]
 keys=['final_best','final_regret','firsthit_score81','actual_evaluations','candidate_none_fraction','median_FM_gap','raw_feasible_fraction','generation_seconds','training_seconds']
 for method in P['methods']:
  rs=[r for r in rows if r['method']==method];groups.append({'method':method,'successes':sum(r['success'] for r in rs),'runs':len(rs),'statistics':{k:trip([r[k] for r in rs]) for k in keys}})
 tests=[]
 for reference in ['XY-p1','LargePenalty-alpha1000']:
  for key in ['final_regret','firsthit_score81','candidate_none_fraction','median_FM_gap']:
   diff=[];seeds=[]
   for seed in P['seeds']:
    a=next(r for r in rows if r['method']=='XY-p3' and r['seed']==seed)[key];b=next(r for r in rows if r['method']==reference and r['seed']==seed)[key]
    if a is not None and b is not None:diff.append(float(a-b));seeds.append(seed)
   p=float(wilcoxon(diff,alternative='two-sided',method='auto').pvalue) if any(abs(v)>1e-12 for v in diff) else 1.;tests.append({'method':'XY-p3','reference':reference,'metric':key,'seeds':seeds,'paired_differences':diff,'p_raw':p,'p_bonferroni':min(1.,8*p)})
 s={'status':'completed','completed_runs':15,'cycles':models,'XY_angle_evaluations':angles,'independent_selected_state_checks':selectedchecks,'shots':77000,'SA_outputs':38500,'f_opt':opt,'f_worst':worst,'groups':groups,'per_seed':rows,'paired_tests':tests};save(OUT/'json/summary.json',s);save(OUT/'json/artifact_verification.json',{'status':'passed','models':1155,'runs':15,'XY_saved_angles_regenerated':98560,'XY_independent_best_state_checks':770,'raw_candidates_verified':115500,'failed_or_dropped':0,'checks':'source/protocol/checkpoint/model SHA,all FM/QUBO predictions,commoninitial3,raw filters/selection/truth/BB budgets,angles regenerated andbestminchecked,chosenstate independentreference,CDF shots,SAalpha1000lambda=Salpha'});report(s);plot(s);print(json.dumps({'groups':groups,'tests':tests},indent=2))

def fmt(t):return '未定義' if t['median'] is None else f"{t['median']:.6g} [{t['q1']:.6g},{t['q3']:.6g}]"

def report(s):
 lines=['# 初期3件でp3再学習最適化','','ユーザー指定:初期3件、XYp3、LargePenaltyはalpha1000。比較用XYp1も同じ3件と設定で再計算。初期3＋77cycles、上限80、100outputs、rank1 AdamW wd0.01 lr0.1 120epochs。5Seed×3手法＝15Run完了。全Seed初期最適解なし。値は5Seed中央値[IQR]。','','| 手法 | 成功 | 最終best eV | 最適解到達評価数/未到達81 | 実評価数 | 候補なし率 | FM gap eV | Raw制約率 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for g in s['groups']:
  st=g['statistics'];lines.append('| '+g['method']+f" | {g['successes']}/5 | "+' | '.join(fmt(st[k]) for k in ['final_best','firsthit_score81','actual_evaluations','candidate_none_fraction','median_FM_gap','raw_feasible_fraction'])+' |')
 lines+=['','FM gapは各cycleの採用候補と未評価実行可能FM最小値との差、各Runの採用cycle中央値を5Seedで集計。81は未到達コードで、81件評価した意味ではない。予算打切りを収束と呼ばない。','','## 各Seed','','| Seed | 手法 | best eV | 到達score | 実評価数 |','|---:|---|---:|---:|---:|']
 for r in s['per_seed']:lines.append(f"| {r['seed']} | {r['method']} | {r['final_best']} | {r['firsthit_score81']} | {r['actual_evaluations']} |")
 lines+=['','## 計算予算','','XYは各cycle128理想期待値角度評価。両深さのstate評価件数は同じだが、p3の層数は3倍。元23One-Hot、sim8bits、lambda0、W/Ring。p3にはp1旧9点をゼロ層で埋め込んだ9点＋119連続ランダム点を使い、補助p1/p2探索は加えない。固定FM段階別warmstart実験とは探索配分が違う。SAは100reads、1000sweeps、自動beta、alpha1000固定、lambda=Salpha。','','| 手法 | 候補生成秒/Run | 学習秒/Run |','|---|---:|---:|']
 for g in s['groups']:lines.append(f"| {g['method']} | {fmt(g['statistics']['generation_seconds'])} | {fmt(g['statistics']['training_seconds'])} |")
 lines+=['','## ペア検定','','p3対p1／SAalpha1000、4指標×2比較＝8Wilcoxon両側、Bonferroni8、同じ5Seedを対応。全差0はp1。','','| 比較基準 | 指標 | n | 差の中央値 | 補正p |','|---|---|---:|---:|---:|']
 for t in s['paired_tests']:lines.append(f"| {t['reference']} | {t['metric']} | {len(t['seeds'])} | {np.median(t['paired_differences']):.6g} | {t['p_bonferroni']:.6g} |")
 (OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')

def plot(s):
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});colors={'XY-p1':'#2ca02c','XY-p3':'#008080','LargePenalty-alpha1000':'#ff7f0e'};markers={'XY-p1':'D','XY-p3':'D','LargePenalty-alpha1000':'s'};styles={'XY-p1':'-','XY-p3':':','LargePenalty-alpha1000':'--'};files=[]
 specs=[('best_evaluations','Best objective (eV)'),('firsthit_score81','First hit score (81 = no hit)'),('final_regret','Final normalized regret'),('median_FM_gap','Accepted unseen FM gap (eV)'),('raw_feasible_fraction','Raw feasible fraction'),('generation_seconds','Candidate generation time per run (s)')]
 def draw(ax,key,label):
  if key in ['best_evaluations','best_cycles']:
   for method in P['methods']:
    rs=[r for r in s['per_seed'] if r['method']==method]
    if key=='best_cycles':domain=np.arange(78);vals=np.array([r['best_by_cycle'] for r in rs]);ax.set_xlabel('Optimization cycle (initial = 0)')
    else:
     end=min(r['actual_evaluations'] for r in rs);domain=np.arange(3,end+1);curves=[]
     for r in rs:
      pairs=dict(r['best_by_evaluation']);curves.append([pairs[n] for n in domain])
     vals=np.array(curves);ax.set_xlabel('True objective evaluations (including initial 3)')
    med=np.median(vals,axis=0);lo,hi=np.percentile(vals,[25,75],axis=0);ax.plot(domain,med,color=colors[method],linestyle=styles[method],label=method,drawstyle='steps-post');ax.fill_between(domain,lo,hi,color=colors[method],alpha=.12,step='post')
   ax.set_ylabel(label)
  else:
   for i,g in enumerate(s['groups']):
    method=g['method'];t=g['statistics'][key];ax.errorbar(i,t['median'],yerr=np.array([[t['median']-t['q1']],[t['q3']-t['median']]]),fmt=markers[method],color=colors[method],capsize=5,label=method)
   ax.set_xticks(range(3),['XY p=1','XY p=3','SA alpha=1000']);ax.set_ylabel(label)
   if key=='raw_feasible_fraction':ax.set_ylim(0,1.05)
   if key=='final_regret':ax.set_ylim(0,max(.05,max(g['statistics'][key]['q3'] for g in s['groups'])*1.1))
  ax.grid(alpha=.2);ax.legend(fontsize=8)
 def export(fig,key):
  for ext in ['pdf','svg','png']:
   path=OUT/ext/f'{key}.{ext}';fig.savefig(path,dpi=300,bbox_inches='tight');files.append({'path':str(path.relative_to(OUT)),'sha256':sha(path)})
 fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
 for ax,(key,label) in zip(axs.flat,specs):draw(ax,key,label)
 export(fig,'initial3_p3_BBO_overview');plt.close(fig)
 for key,label in specs+[('best_cycles','Best objective (eV)')]:
  fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained');draw(ax,key,label);export(fig,key);plt.close(fig)
 save(OUT/'json/figure_manifest.json',{'summary_sha256':sha(OUT/'json/summary.json'),'files':files})

if __name__=='__main__':main()
