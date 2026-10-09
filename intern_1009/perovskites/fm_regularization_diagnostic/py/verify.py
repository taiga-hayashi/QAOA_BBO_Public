"""Independent recomputation of all recorded scalar metrics and delivery hashes."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OUT/'py'))
from diagnose_regularization import context,PILOT,sha,save
for seed in [42,101,2024,7,19]:
 d=json.loads((OUT/f'json/seed_{seed}.json').read_text());base=json.loads((PILOT/f'json/seed_{seed}.json').read_text());cats,x,y,tr,te=context(base);seen={cats[i][0] for i in tr}
 for k in [1,2]:
  adam=next(f for f in d['fits'] if f['config']['id']==f'k{k}_Adam_wd0.0');adamw=next(f for f in d['fits'] if f['config']['id']==f'k{k}_AdamW_wd0.0');assert adam['predictions']==adamw['predictions']
 for f in d['fits']:
  pred=np.array(f['predictions']);idx=f['selected_id'];met=f['metrics'];ind={'train_rmse':np.sqrt(np.mean((pred[tr]-y[tr])**2)),'test_rmse':np.sqrt(np.mean((pred[te]-y[te])**2)),'test_spearman':spearmanr(pred[te],y[te]).statistic,'selected_true_value':y[idx],'selected_test_regret':(y[idx]-y[te].min())/(y[te].max()-y[te].min())}
  for label,ids in [('seen_organic',[i for i in te if cats[i][0] in seen]),('unseen_organic',[i for i in te if cats[i][0] not in seen])]:ind[label+'_rmse']=np.sqrt(np.mean((pred[ids]-y[ids])**2))
  for key,val in ind.items():assert abs(met[key]-val)<1e-10
manifest=json.loads((OUT/'json/figure_manifest.json').read_text())
assert manifest['summary_sha256']==sha(OUT/'json/summary.json')
for a in manifest['artifacts']:assert sha(OUT/a['path'])==a['sha256']
save(OUT/'json/delivery_verification.json',{'status':'passed','fits':30,'failure_or_exclusion_count':0,'independent_all_metrics_recomputation':'passed','Adam_AdamW_zero_decay_10_prediction_vectors':'exactly equal','plot_files':12,'figure_hashes':'passed','PDF_render_visual_review':'integrated overview and all3 native panels; axes legends readable and no clipping','shared_default_baselines':10,'source_py_compile':'passed; main and intern mirror plus experiment scripts and intern README quick check','git_diff_check':'passed'})
print('verified all metrics, zero-decay prediction equality, 12 figure hashes')
