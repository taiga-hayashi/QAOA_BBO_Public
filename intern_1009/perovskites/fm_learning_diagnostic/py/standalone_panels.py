"""Standalone renderings from the verified diagnostic summary."""
from diagnose import OUT
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
result=json.loads((OUT/'json/summary.json').read_text())
for key,label,name in [('test_rmse','Test RMSE (eV)','test_rmse'),('test_spearman','Test Spearman','test_spearman'),('selected_true_value','True value of predicted best unseen (eV)','selected_true_value')]:
 fig,ax=plt.subplots(figsize=(5.8,4.3),layout='constrained')
 for lr,ep,color,marker in [(0.1,120,'#1f77b4','o'),(0.01,1000,'#ff7f0e','s')]:
  gs=[next(g for g in result['groups'] if g['config']['id']==f'k{k}_lr{lr}_e{ep}_raw')['statistics'][key] for k in [0,1,2,4]]
  val=lambda name:np.array([g[name] for g in gs]);ax.plot([0,1,2,4],val('median'),color=color,marker=marker,label=f'lr={lr}, epochs={ep}');ax.fill_between([0,1,2,4],val('q1'),val('q3'),color=color,alpha=.12)
 ax.set_xticks([0,1,2,4]);ax.set_xlabel('FM rank (0: additive)');ax.set_ylabel(label);ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
 for ext in ['pdf','png','svg']:fig.savefig(OUT/f'{ext}/{name}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
