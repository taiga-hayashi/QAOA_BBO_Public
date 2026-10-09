"""Independent checkpoint audit and matched four-policy BBO report/plots."""
import sys,json
import numpy as np
import torch
from scipy.stats import wilcoxon
from run import OUT,ROOT,BASE,P,FILES,REPO,sha,save,initial,problem,model_hash,pick
sys.path.insert(0,str(ROOT/'xy_p3_initial3_bbo/py'))
from analyze_initial3 import TorchFM,trip,plt
from run_bbo import fm_to_qubo,matrix
METHODS=['Exact-FM','Random-unseen','XY-p3','LargePenalty-alpha1000']

def main():
 v=json.loads((OUT/'json/prevalidation.json').read_text());assert v['status']=='passed' and v['source_sha256']=={str(c.relative_to(REPO)):sha(c) for c in FILES}
 bb,cats,x=problem();truth=np.array([bb.evaluate(c) for c in cats]);opt=truth.min();worst=truth.max();runs=[];newmodels=0
 for item in P['baseline_files']:assert sha(ROOT/item['path'])==item['sha256']
 oldaudit=json.loads((BASE/'json/artifact_verification.json').read_text());assert oldaudit['status']=='passed'
 for method in METHODS:
  folder=OUT if method in P['methods'] else BASE
  for seed in P['seeds']:
   d=json.loads((folder/f'json/{method}_seed{seed}.json').read_text());assert d['status']=='completed' and len(d['events'])==77 and d['initial_ids']==initial(seed);assert sha(folder/d['checkpoint'])==d['checkpoint_sha256'];ids=initial(seed);vals=truth[ids].tolist();assert vals==d['initial_values'];curve=[min(vals)];hit=None;states=torch.load(folder/d['checkpoint'],weights_only=False) if folder==OUT else None
   for index,e in enumerate(d['events']):
    cycle=index+1;assert e['cycle']==cycle and e['evaluations_before']==len(ids) and e['train_ids_before']==ids
    if folder==OUT:
     cp=states[index];assert cp['cycle']==cycle;m=TorchFM(23,1);m.load_state_dict(cp['state_dict']);assert model_hash(m)==cp['model_hash']==e['model_sha256'];qd,b=fm_to_qubo(m);q=matrix(qd,23);pred=np.einsum('bi,ij,bj->b',x,q,x)+b;np.testing.assert_allclose(pred,e['fm_predictions'],atol=1e-10,rtol=0)
     with torch.no_grad():np.testing.assert_allclose(pred,m(torch.tensor(x,dtype=torch.float32)).numpy(),atol=1e-4,rtol=0)
     assert pick(method,pred,x,ids,seed,cycle)==e['selected_id'] and e['model_seed']==seed+100000*(cycle-1);newmodels+=1
     if cycle==1:assert e['model_sha256']==next(z for z in v['checks'] if z['seed']==seed)['model_sha256']
    i=e['selected_id'];assert i is not None and i not in ids and e['selected_value']==truth[i];ids.append(i);vals.append(e['selected_value']);assert e['evaluations_after']==len(ids) and e['best_so_far']==min(vals);curve.append(min(vals))
    if hit is None and min(vals)==opt:hit=len(ids)
   assert len(ids)==len(set(ids))==d['actual_evaluations']==80 and ids==d['final_ids'] and vals==d['final_values'] and min(vals)==d['final_best']
   runs.append({'method':method,'seed':seed,'final_best':min(vals),'final_regret':float((min(vals)-opt)/(worst-opt)),'firsthit_score81':hit if hit is not None else 81,'success':hit is not None,'best_by_evaluation':curve,'generation_seconds':sum(e['generation_seconds'] for e in d['events']),'training_seconds':sum(e['training_seconds'] for e in d['events'])})
 assert newmodels==770
 tests=[]
 for reference in METHODS[1:]:
  for metric in ['final_regret','firsthit_score81']:
   delta=[next(r for r in runs if r['method']=='Exact-FM' and r['seed']==seed)[metric]-next(r for r in runs if r['method']==reference and r['seed']==seed)[metric] for seed in P['seeds']];pv=float(wilcoxon(delta).pvalue) if any(abs(z)>1e-12 for z in delta) else 1.;tests.append({'reference':reference,'metric':metric,'paired_differences':delta,'p_raw':pv,'p_bonferroni':min(1,6*pv)})
 s={'status':'completed','new_runs':10,'new_models_audited':770,'baseline_runs_reused':10,'per_seed':runs,'paired_tests':tests,'protocol_sha256':sha(OUT/'json/protocol.json')};save(OUT/'json/summary.json',s)
 lines=['# Exact-FM／ランダムBBO対照','','新規10Runと保存済みXY-p3／SAalpha1000の10Runを同じ初期3件・5Seed・77サイクル・最大80評価で比較。全列挙は真値を使わず、未評価候補のFM予測値のみで選ぶ。Randomは未評価集合から一様に1件を選ぶ。','','| 手法 | 成功 | 最終best eV中央値[IQR] | 到達評価数中央値[IQR]（未到達81） |','|---|---:|---:|---:|']
 def fmt(t):return f"{t['median']:.6g} [{t['q1']:.6g}, {t['q3']:.6g}]"
 for method in METHODS:
  rr=[r for r in runs if r['method']==method];lines.append(f"| {method} | {sum(r['success'] for r in rr)}/5 | {fmt(trip([r['final_best'] for r in rr]))} | {fmt(trip([r['firsthit_score81'] for r in rr]))} |")
 lines+=['','## 全Seed','','| 手法 | Seed | 最終best | 到達score |','|---|---:|---:|---:|']
 for r in runs:lines.append(f"| {r['method']} | {r['seed']} | {r['final_best']} | {r['firsthit_score81']} |")
 lines+=['','## 対応Wilcoxon検定（Bonferroni6）','','| Exact-FMの対照 | 指標 | 補正p |','|---|---|---:|']
 for t in tests:lines.append(f"| {t['reference']} | {t['metric']} | {t['p_bonferroni']:.6g} |")
 lines+=['','81は未到達コードで、実評価数は全Run80。初期3件込み。最適値取得は全192真値の事後照合で判定。全列挙はサンプリング100出力と計算量を揃えた手法ではなく、候補選択の診断用対照。RandomではFMを学習・保存するが選択には使わない。純ランダム探索に必要な処理時間はその学習を除く。失敗Runは削除せず計算失敗で後処理を停止する。補正非有意は同等性の証明ではない。']
 (OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')
 colors=['#9467bd','#7f7f7f','#008080','#ff7f0e'];styles=['-','-.',':','--'];specs=[('curve','Best objective (eV)'),('firsthit_score81','First hit score (81 = no hit)'),('final_regret','Final normalized regret'),('generation_seconds','Candidate selection time per run (s)')];files=[]
 def draw(ax,key,label):
  for idx,(method,col,style) in enumerate(zip(METHODS,colors,styles)):
   rr=[r for r in runs if r['method']==method]
   if key=='curve':
    a=np.array([r['best_by_evaluation'] for r in rr]);lo,med,hi=np.percentile(a,[25,50,75],axis=0);domain=np.arange(3,81);ax.plot(domain,med,color=col,linestyle=style,label=method,drawstyle='steps-post');ax.fill_between(domain,lo,hi,color=col,alpha=.12,step='post');ax.set_xlabel('True evaluations (including initial 3)')
   else:
    t=trip([r[key] for r in rr]);ax.errorbar(idx,t['median'],yerr=[[t['median']-t['q1']],[t['q3']-t['median']]],fmt=['o','x','D','s'][idx],color=col,capsize=4,label=method);ax.set_xticks(range(4),['Exact FM','Random','XY p=3','SA alpha=1000'])
  ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=8)
 def export(fig,name):
  for ext in ['pdf','svg','png']:
   path=OUT/ext/f'{name}.{ext}';fig.savefig(path,dpi=300,bbox_inches='tight');files.append({'path':str(path.relative_to(OUT)),'sha256':sha(path)})
 fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
 for ax,(key,label) in zip(axes.flat,specs):draw(ax,key,label)
 export(fig,'exact_random_BBO_overview');plt.close(fig)
 for key,label in specs:
  fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');draw(ax,key,label);export(fig,key);plt.close(fig)
 save(OUT/'json/figure_manifest.json',{'summary_sha256':sha(OUT/'json/summary.json'),'files':files,'visual_QA':'pending next interactive review'});save(OUT/'json/artifact_verification.json',{'status':'passed','new_checkpoint_models_checked':770,'baseline_runs_hash_verified':10,'raw_BBO_budget_and_selection':'passed','visual_QA':'pending'});print('Audited and plotted; visual QA pending.',flush=True)

if __name__=='__main__':torch.set_num_threads(1);main()
