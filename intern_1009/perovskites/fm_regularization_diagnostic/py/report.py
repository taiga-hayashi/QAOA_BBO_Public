"""Render measured median/IQR results; generate integrated and native panels."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OUT.parent/'fixed_fm_pilot/py'))
from run_pilot import save,sha
S=json.loads((OUT/'json/summary.json').read_text())
METRICS=[('test_rmse','Test RMSE (eV)'),('test_spearman','Test Spearman'),('selected_true_value','True value of predicted best unseen (eV)')]

def panel(ax,key,label):
 for k,color,marker,shift in [(1,'#0072B2','o',-.04),(2,'#D55E00','s',.04)]:
  gs=[next(g for g in S['groups'] if g['config']['id']==f'k{k}_{opt}_wd{wd}')['statistics'][key] for opt,wd in [('Adam',0.),('AdamW',0.),('AdamW',.01)]]
  v=lambda name:np.array([g[name] for g in gs]);ax.errorbar(np.arange(3)+shift,v('median'),yerr=np.array([v('median')-v('q1'),v('q3')-v('median')]),color=color,marker=marker,capsize=4,label=f'Rank {k} (median / IQR)',linewidth=1.5)
 ax.set_xticks(range(3),['Adam\nwd = 0','AdamW\nwd = 0','AdamW\nwd = 0.01']);ax.set_xlabel('Optimizer');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)

fig,axs=plt.subplots(1,3,figsize=(14.5,4.5),layout='constrained')
for ax,(key,label) in zip(axs,METRICS):panel(ax,key,label)
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{ext}/fm_regularization_overview.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)
for key,label in METRICS:
 fig,ax=plt.subplots(figsize=(6.6,4.5),layout='constrained');panel(ax,key,label)
 for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{ext}/{key}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
lines=['# Adam / AdamW・weight decay比較','','初期20件・5 Seedを保持し、rank1/2 × Adam wd0 / AdamW wd0 / AdamW wd0.01を比較した。lr0.1、120 epochs、raw eV、全バッチMSE。30学習を完了し、失敗・除外は0。以下は中央値 [Q1,Q3]。','','| Rank | Optimizer | wd | Train RMSE eV | Test RMSE eV | Test Spearman | 予測最小の未評価候補の真値 eV |','|---:|---|---:|---:|---:|---:|---:|']
for g in S['groups']:
 c=g['config'];fmt=lambda key:'{median:.4f} [{q1:.4f}, {q3:.4f}]'.format(**g['statistics'][key]);lines.append(f"| {c['rank']} | {c['optimizer']} | {c['weight_decay']} | "+' | '.join(fmt(key) for key in ['train_rmse','test_rmse','test_spearman','selected_true_value'])+' |')
lines+=['','## 同一Seedのペア差分','','差分は前者−後者。RMSEと真値は負、相関は正で改善。Wilcoxon検定は事前固定した18比較をBonferroni補正。','','| Rank | 比較 | 指標 | 差分中央値 | raw p | 補正p |','|---:|---|---|---:|---:|---:|']
for t in S['paired_tests']:lines.append(f"| {t['rank']} | {t['contrast']} | {t['metric']} | {t['median_difference']:.5g} | {t['p_raw']:.5g} | {t['p_bonferroni']:.5g} |")
lines+=['','Seed別の全値はjson/seed_*.json、集計と5個のペア差分はjson/summary.jsonに保存。初期20件を除く172候補で評価し、予測最小候補は真値を使わず全列挙で選択する。真値は保存済みlookup表を事後参照した値。selected_test_regretの基準は各Seedの172候補のmin/maxであり、全192候補のRegretと異なる。','', 'この比較では既に見たtestを再使用している。探索的診断であり、採用設定の独立確認ではない。BBO・SA・QAOA・新規物理評価は実施していない。']
(OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')
(OUT/'md/FIGURES.md').write_text('# 図の見方\n\n[全体PDF](../pdf/fm_regularization_overview.pdf)の横軸はOptimizerとweight decay。色はランク1/2、点は5 Seedの中央値、エラーバーはIQR。左と右は低い方、中央は高い方が良い。右は各SeedでFMが最良と予測した未評価候補の真値で、BBO best-so-farではない。\n\n単独図test_rmse / test_spearman / selected_true_valueを含め、PDF/SVG/PNGの計12ファイル。タイトル・下部注記は付けず、凡例・軸で表現した。\n')
artifacts=[{'path':str(p.relative_to(OUT)),'sha256':sha(p),'bytes':p.stat().st_size} for ext in ['pdf','svg','png'] for p in sorted((OUT/ext).glob('*.'+ext))]
assert len(artifacts)==12
save(OUT/'json/figure_manifest.json',{'status':'generated','summary_sha256':sha(OUT/'json/summary.json'),'report_script_sha256':sha(Path(__file__)),'artifacts':artifacts})
print('generated 12 figures and results')
