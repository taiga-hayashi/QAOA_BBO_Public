"""Independent audit of every angle and measurement; single-model p comparison."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from run_depth import OUT,ROOT,REPO,P,CODE,frozen,reference_layers,angles,sha,save,select,model_hash
import sys
sys.path.insert(0,str(ROOT/'large_penalty_diagnostic/py'))
from analyze_penalty import trip,plt

def main():
 v=json.loads((OUT/'json/prevalidation.json').read_text());assert v['status']=='passed' and v['source_sha256']=={str(c.relative_to(REPO)):sha(c) for c in CODE};m,q,b,x,pred,s,ids=frozen();div=s if s>0 else 1.;order=np.argsort(x.astype(np.uint64)@(1<<np.arange(23,dtype=np.uint64)));unseen=[i for i in range(192) if i not in ids];umin=float(pred[unseen].min());minimum=[i for i in unseen if abs(pred[i]-umin)<1e-8];rows=[];count=0
 for seed in range(10):
  previous=None
  for p in P['depths']:
   d=json.loads((OUT/f'json/searchseed{seed}_p{p}.json').read_text());assert d['status']=='completed' and d['p']==p and d['search_seed']==seed and d['lambda']==0 and d['model_sha256']==model_hash(m) and d['protocol_sha256']==sha(OUT/'json/protocol.json') and len(d['search'])==128 and len(d['measurements'])==10;assert d['S']==s and d['normalization_divisor']==div
   expected=angles(p,seed,previous);reference_probs=[]
   for e,(gs,bs) in zip(d['search'],expected):
    assert e['gammas']==gs and e['betas']==bs;state=reference_layers(q/div,[16,3,4],gs,bs);prob=abs(state)**2;value=float(prob@pred);assert abs(value-e['expected_FM'])<1e-10 and abs(1-prob.sum())<1e-10 and abs(1-e['feasible_mass'])<1e-10;reference_probs.append(prob);count+=1
   best=min(d['search'],key=lambda e:e['expected_FM']);assert best==d['best_angles'];prob=np.array(d['probabilities']);np.testing.assert_allclose(prob,reference_probs[best['index']],atol=1e-10,rtol=0);assert prob.shape==(192,) and np.isfinite(prob).all() and (prob>=0).all();cdf=np.cumsum(prob[order]);mhit=[];gaps=[];none=[];unique=[];pool=[]
   for rep,e in enumerate(d['measurements']):
    assert e['rep']==rep;rng=np.random.default_rng(np.random.SeedSequence([42,1009,seed,rep,93]));draws=order[np.searchsorted(cdf,rng.random(100)*cdf[-1],side='left')];np.testing.assert_array_equal(draws,e['raw_candidate_ids']);i,f=select(x[draws],pred,x,set(ids));assert i==e['selected_id']
    for k,val in f.items():assert val==e[k]
    gap=float(pred[i]-umin) if i is not None else None;assert gap==e['selected_unseen_gap'];hit=bool(any(i in minimum for i in draws));assert hit==e['minimum_hit'];assert f['raw_feasible_rate']==1.;mhit.append(hit);gaps.append(gap);none.append(i is None);unique.append(f['unique_unseen_count']);pool.extend(pred[draws[[i not in ids for i in draws]]]-umin)
   mass=float(prob[minimum].sum());assert abs(mass-d['ideal_minimum_mass'])<1e-12 and abs(d['ideal_100shot_minimum_capture']-(1-(1-mass)**100))<1e-12;assert abs(d['ideal_unseen_mass']-prob[unseen].sum())<1e-12;assert abs(d['expected_FM_gap_to_global_min']-(float(prob@pred)-pred.min()))<1e-10
   if previous:assert best['expected_FM']<=previous['expected_FM']+1e-10
   rows.append({'search_seed':seed,'p':p,'expected_gap':d['expected_FM_gap_to_global_min'],'ideal_minimum_mass':mass,'ideal_100shot_capture':d['ideal_100shot_minimum_capture'],'empirical_capture_fraction':float(np.mean(mhit)),'capture_count':sum(mhit),'median_selected_gap':trip(gaps)['median'],'selected_gap_stats':trip(gaps),'candidate_none_fraction':float(np.mean(none)),'mean_raw_unseen_gap':float(np.mean(pool)) if pool else None,'median_unique_unseen':trip(unique)['median'],'total_seconds':d['total_seconds'],'search_seconds':sum(e['seconds'] for e in d['search']),'best_warm_start':p>1 and best['index']==0,'best_angles':best,'probabilities':prob.tolist(),'best_sofar_search':np.minimum.accumulate([e['expected_FM']-pred.min() for e in d['search']]).tolist()});previous=best
 assert count==3840;groups=[]
 for p in P['depths']:
  rs=[r for r in rows if r['p']==p];keys=['expected_gap','ideal_minimum_mass','ideal_100shot_capture','empirical_capture_fraction','median_selected_gap','candidate_none_fraction','mean_raw_unseen_gap','median_unique_unseen','total_seconds','search_seconds'];groups.append({'p':p,'statistics':{k:trip([r[k] for r in rs]) for k in keys},'total_empirical_hits':sum(r['capture_count'] for r in rs),'total_measurement_batches':100,'warm_start_selected_count':sum(bool(r['best_warm_start']) for r in rs),'state_evaluations':1280,'search_layer_evaluations':1280*p})
 tests=[]
 for p in [2,3]:
  for key in ['expected_gap','median_selected_gap','empirical_capture_fraction']:
   diff=[];ss=[]
   for seed in range(10):
    a=next(r for r in rows if r['p']==p and r['search_seed']==seed)[key];b=next(r for r in rows if r['p']==1 and r['search_seed']==seed)[key]
    if a is not None and b is not None:diff.append(float(a-b));ss.append(seed)
   pv=float(wilcoxon(diff,alternative='two-sided',method='auto').pvalue) if any(abs(z)>1e-12 for z in diff) else 1.;tests.append({'p':p,'reference':1,'metric':key,'search_seeds':ss,'differences':diff,'p_raw':pv,'p_bonferroni':min(1.,6*pv)})
 summary={'status':'completed','model_count':1,'model_seed':42,'model_hash':model_hash(m),'initial_ids':ids,'FM_predictions':pred.tolist(),'global_FM_minimum':float(pred.min()),'unseen_FM_minimum':umin,'unseen_minimum_ids':minimum,'groups':groups,'per_search_seed':rows,'paired_tests':tests,'old9_p1':v['old9_p1_best']};save(OUT/'json/summary.json',summary);save(OUT/'json/artifact_verification.json',{'status':'passed','independent_angle_checks':3840,'measurement_batches':300,'shots_verified':30000,'completed_searches':30,'failed_or_dropped':0,'checks':'source/protocol/frozenFM hashes,all candidateangles and expectedenergies/feasibility,chosen probabilities independently,all CDFdraws/selection/gaps,exact ideal mass/capture formula,warm-start monotonic search objective'});report(summary);plot(summary);print(json.dumps({'groups':groups,'tests':tests},indent=2))

def fmt(v):return '未定義' if v['median'] is None else f"{v['median']:.6g} [{v['q1']:.6g},{v['q3']:.6g}]"

def report(s):
 lines=['# 固定FMに対するXY-QAOA深さ比較','','Seed42・初期10から学習した1つのFM、lambda0、W/Ring。元23One-Hot変数、厳密符号化simulator8bits。p1/2/3、128角度候補×10探索Seed、最終100shots×10測定反復/探索Seed。30探索・3840角度評価・300測定batchesを完了。新規FM学習・BBOなし。中央値[IQR]は10探索Seed間。','','| p | 理想期待FM gap eV | 理想100shot最小捕捉率 | 経験捕捉率(探索内頻度) | 選択gap(探索内中央値) eV | 全測定捕捉 | warm解選択 |','|---:|---:|---:|---:|---:|---:|---:|']
 for g in s['groups']:
  st=g['statistics'];lines.append(f"| {g['p']} | {fmt(st['expected_gap'])} | {fmt(st['ideal_100shot_capture'])} | {fmt(st['empirical_capture_fraction'])} | {fmt(st['median_selected_gap'])} | {g['total_empirical_hits']}/100 | {g['warm_start_selected_count']}/10 |")
 lines+=['','理想期待gapは全実行可能FM最小値との差。最小捕捉と選択gapは既評価10点を除く未評価最小値が基準。今回は全FM最小と未評価FM最小が同じ。理想捕捉率は理想IID shotsとして1-(1-Pmin)^100を厳密確率から計算。経験率は100shotsの測定100反復の観測回数で、モデル間成功率ではない。選択gapは各探索Seed内10測定の中央値を取り、探索Seed間中央値/IQR。','','## 探索・計算予算','','全pで1280角度state評価。総層評価はp1:1280、p2:2560、p3:3840。別途best状態の再計算10件/p。各層gamma,betaは0〜0.8、p1には旧9gridを含め、p2/3には前深さbestのゼロ層埋込みを1候補として含める。期待値非悪化はこのwarm候補の含有で保証され、自然に高pがよいという独立の証拠ではない。旧p1の9gridベストは'+str(s['old9_p1']['expected_FM'])+' eV（期待値）。','','| p | raw未評価平均gap eV | 未評価ユニーク数中央値 | 探索秒 |','|---:|---:|---:|---:|']
 for g in s['groups']:lines.append(f"| {g['p']} | {fmt(g['statistics']['mean_raw_unseen_gap'])} | {fmt(g['statistics']['median_unique_unseen'])} | {fmt(g['statistics']['search_seconds'])} |")
 lines+=['','## 全探索Seed','','| Seed | p | 期待gap eV | 理想100shot捕捉 | 観測捕捉/10 | 選択gap中央値 eV |','|---:|---:|---:|---:|---:|---:|']
 for r in s['per_search_seed']:lines.append(f"| {r['search_seed']} | {r['p']} | {r['expected_gap']:.6g} | {r['ideal_100shot_capture']:.6g} | {r['capture_count']}/10 | {r['median_selected_gap']} |")
 lines+=['','## 対応検定','','p2/3対1、3指標、6Wilcoxon、Bonferroni6。この1FMと探索Seed分布に条件付けた検定。全差0はp1。','','| p | 指標 | n | ペア差中央値 | 補正p |','|---:|---|---:|---:|---:|']
 for t in s['paired_tests']:lines.append(f"| {t['p']} | {t['metric']} | {len(t['differences'])} | {np.median(t['differences']):.6g} | {t['p_bonferroni']:.6g} |")
 (OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')

def plot(s):
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});gs=s['groups'];ps=np.array(P['depths']);colors=['#2ca02c','#008080','#9467bd'];files=[]
 specs=[('expected_gap','Ideal expected FM gap (eV)'),('ideal_100shot_capture','Ideal 100-shot minimum capture'),('median_selected_gap','Selected unseen FM gap (eV)'),('empirical_capture_fraction','Empirical 100-shot capture frequency')]
 def draw(ax,key):
  if key=='gap_CDF':
   for p,c in zip(ps,colors):
    rs=[r for r in s['per_search_seed'] if r['p']==p];ids=[i for i in range(192) if i not in s['initial_ids']];gap=np.array(s['FM_predictions'])[ids]-s['unseen_FM_minimum'];order=np.argsort(gap);probs=np.array([np.array(r['probabilities'])[ids]/sum(np.array(r['probabilities'])[ids]) for r in rs]);cdfs=np.cumsum(probs[:,order],axis=1);med=np.median(cdfs,axis=0);lo,hi=np.percentile(cdfs,[25,75],axis=0);ax.step(gap[order],med,where='post',color=c,label=f'p={p}');ax.fill_between(gap[order],lo,hi,color=c,alpha=.15,step='post')
   ax.set_xlabel('Unseen FM gap (eV)');ax.set_ylabel('Ideal conditional cumulative mass')
  elif key=='angle_search':
   for p,c in zip(ps,colors):
    vals=np.array([r['best_sofar_search'] for r in s['per_search_seed'] if r['p']==p]);med=np.median(vals,axis=0);lo,hi=np.percentile(vals,[25,75],axis=0);ax.plot(range(1,129),med,color=c,label=f'p={p}');ax.fill_between(range(1,129),lo,hi,color=c,alpha=.15)
   ax.set_xlabel('Angle candidate evaluations');ax.set_ylabel('Best ideal expected FM gap (eV)')
  else:
   label=dict(specs)[key];ts=[g['statistics'][key] for g in gs];med=np.array([t['median'] for t in ts]);ax.errorbar(ps,med,yerr=np.array([med-np.array([t['q1'] for t in ts]),np.array([t['q3'] for t in ts])-med]),fmt='D-',capsize=4,color='#2ca02c',label='XY, search-seed median / IQR');ax.set_xlabel('QAOA depth p');ax.set_xticks(ps);ax.set_ylabel(label)
   if 'capture' in key:ax.set_ylim(0,1.05)
  ax.grid(alpha=.2);ax.legend(fontsize=8)
 def export(fig,key):
  for ext in ['pdf','svg','png']:
   path=OUT/ext/f'{key}.{ext}';fig.savefig(path,dpi=300,bbox_inches='tight');files.append({'path':str(path.relative_to(OUT)),'sha256':sha(path)})
 keys=[k for k,_ in specs]+['gap_CDF','angle_search'];fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
 for ax,k in zip(axs.flat,keys):draw(ax,k)
 export(fig,'XY_depth_overview');plt.close(fig)
 for k in keys:
  fig,ax=plt.subplots(figsize=(6.6,4.5),layout='constrained');draw(ax,k);export(fig,k);plt.close(fig)
 save(OUT/'json/figure_manifest.json',{'summary_sha256':sha(OUT/'json/summary.json'),'files':files})

if __name__=='__main__':main()
