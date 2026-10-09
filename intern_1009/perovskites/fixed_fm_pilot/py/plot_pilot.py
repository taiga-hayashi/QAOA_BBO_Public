"""Standalone scientific figures from saved pilot records only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parents[1]
COLORS=['#1f77b4','#ff7f0e','#d62728','#2ca02c']
MARKERS=['o','s','^','D']
LABELS=['Adaptive\nSA','Large-penalty\nSA','Penalty\nQAOA','XY\nQAOA']


def panel(ax,summary,methods,key,label,log=False,limits=None):
    offsets=np.linspace(-0.065,0.065,len(summary['per_seed']))
    for x,method in enumerate(methods):
        values=[]
        for shift,row in zip(offsets,summary['per_seed']):
            value=row['methods'].get(method,{}).get(key)
            if value is None:continue
            values.append(value)
            ax.scatter(x+shift,value,color=COLORS[x],marker=MARKERS[x],s=25,alpha=0.8,zorder=3)
        if values:
            q1,median,q3=np.quantile(values,[0.25,0.5,0.75])
            ax.errorbar(x,median,yerr=[[median-q1],[q3-median]],fmt='_',markersize=17,color='#202020',capsize=5,lw=1.6,zorder=4)
    ax.set_xticks(range(len(methods)),LABELS)
    ax.set_ylabel(label)
    if log:ax.set_yscale('log')
    if limits:ax.set_ylim(*limits)
    ax.set_xlim(-0.45,len(methods)-0.55)
    ax.grid(axis='y',color='#dddddd',lw=0.6)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(handles=[Line2D([],[],color='#666666',marker='o',ls='',label='Seed value'),Line2D([],[],color='#202020',marker='_',ls='-',label='Median and IQR')],loc='best',fontsize=8,frameon=False)


def export(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(OUT/ext/f'{name}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def main():
    summary=json.loads((OUT/'json/summary.json').read_text())
    protocol=json.loads((OUT/'json/protocol.json').read_text());assert protocol['status']=='ready'
    assert summary['completed_seed_count']==summary['planned_seed_count']
    plt.rcParams.update({'font.size':10,'axes.labelsize':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    methods=protocol['methods']
    specs=[('raw_feasibility','empirical_raw_feasible_rate','Raw feasible fraction',False,(-0.035,1.065)),
           ('true_regret','normalized_regret_initial_plus_proposals','Regret: initial + proposed',False,(-0.005,0.12)),
           ('generation_time','generation_seconds','Candidate generation time (s)',True,None)]
    fig,axes=plt.subplots(1,3,figsize=(12.7,3.7),layout='constrained')
    for ax,(_,key,label,log,limits) in zip(axes,specs):panel(ax,summary,methods,key,label,log,limits)
    export(fig,'fixed_fm_overview')
    for name,key,label,log,limits in specs:
        fig,ax=plt.subplots(figsize=(4.4,3.7),layout='constrained');panel(ax,summary,methods,key,label,log,limits);export(fig,name)
    (OUT/'md/FIGURES.md').write_text('''# 図の読み方

fixed_fm_overviewは3パネル、raw_feasibility・true_regret・generation_timeは独立に描画した個別図。PDF/SVG/PNGを保存。タイトル・下部注釈は付けず、この文書に条件を記載した。

各点は同じ5 Seedの実測値、黒い線は中央値とIQR（Q1〜Q3）。各手法1,000サンプル、初期20件、未評価候補を最大5件採用。無効候補や不足は除外せず、Regretには初期候補の最良値も含める。

Raw feasibleはサンプル中の実行可能割合（線形0〜1）。Regretは監査済みの192候補表の最小・最大で正規化。線形軸で今回の値を読み取れるよう表示範囲を約0〜0.12にした。時間は候補生成の壁時計秒（対数軸）、FM学習・真値診断の時間を含まない。計算資源を同等にした速度比較ではない。

QAOAはOpenQARPの理想23量子ビット・p=1・9点探索・1,000ショット。SAは1,000 reads×1,000 sweeps。固定FMでの診断であり、BBO軌跡や量子優位性の図ではない。全データ・指標・条件はjson/summary.json、json/seed_*.json、json/protocol.jsonにある。
''')


if __name__=='__main__':main()
