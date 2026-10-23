"""Figures for the closed-loop FMQA comparison (experiments/mlfilter-scale, bbo-confirm: L=7, budget 100, 15 seeds).

  python plot.py            rebuild the CSVs from the run, then draw
  python plot.py --replot   draw from csv/ only

Colours, fonts and sizes come from src/aps_style.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT/'src'))
import aps_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CONFIG = json.loads((HERE/'analysis_config.json').read_text())
CSV = HERE/'csv'
ORDER = ['XY-FMQAOA', 'alpha1', 'alpha100', 'alpha0.3', 'RandomFM', 'Random']
LABEL = {'XY-FMQAOA': r'XY-FMQAOA ($\lambda=0$)', 'alpha1': r'SA, $\alpha=1$ (tuned)', 'alpha100': r'SA, $\alpha=100$ (too large)',
         'alpha0.3': r'SA, $\alpha=0.3$ (too small)', 'RandomFM': 'Random proposals + FM selection', 'Random': 'Random search'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_csv():
    run = PROJECT/CONFIG['input']['run_dir']
    meta = json.loads((run/'metadata.json').read_text())
    table = PROJECT/meta['truth_from']['run_dir']/'raw/truth_table.npz'
    assert sha(table)==meta['truth_from']['truth_table_sha256']
    truth = np.load(table)['fom']; order = np.argsort(truth, kind='stable')
    rank = np.empty(len(truth), int); rank[order] = np.arange(1, len(truth)+1)
    B = CONFIG['budget']; curves = []; finals = []
    for f, digest in CONFIG['input']['files_sha256'].items():
        p = run/'raw/bbo'/f; assert sha(p)==digest, f
        d = json.loads(p.read_text()); cid = d['condition']['id']; ids = d['final_ids'][:B]
        best = np.minimum.accumulate(rank[ids])
        if len(best) < B:   # proposal cycles ran out (no new feasible candidate): the best stays as it was, budget unused
            best = np.concatenate([best, np.full(B-len(best), best[-1])])
        for n, r in enumerate(best, 1):
            if n >= 40: curves.append({'condition':cid,'seed':d['seed'],'evaluations':n,'best_rank':int(r)})
        finals.append({'condition':cid,'seed':d['seed'],'best_rank':int(best[-1]),'reached_optimum':int(best[-1]==1),'actual_evaluations':len(ids)})
    CSV.mkdir(exist_ok=True)
    for name, rows in (('best_rank_curves.csv', curves), ('final_per_seed.csv', finals)):
        with open(CSV/name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    agg = []
    for cid in ORDER:
        for n in range(40, B+1):
            v = np.array([r['best_rank'] for r in curves if r['condition']==cid and r['evaluations']==n])
            agg.append({'condition':cid,'evaluations':n,'median':float(np.median(v)),'q25':float(np.percentile(v,25)),
                        'q75':float(np.percentile(v,75)),'fraction_optimum':float(np.mean(v==1)),'n_seeds':len(v)})
    with open(CSV/'best_rank_summary.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, list(agg[0])); w.writeheader(); w.writerows(agg)
    print('CSV written:', ', '.join(p.name for p in sorted(CSV.glob('*.csv'))))


def read(name):
    with open(CSV/name) as fh:
        return [{k:(v if k=='condition' else float(v)) for k, v in r.items()} for r in csv.DictReader(fh)]


def _series(rows, cid, key):
    r = [x for x in rows if x['condition']==cid]
    return np.array([x['evaluations'] for x in r]), np.array([x[key] for x in r])


def _rank_panel(ax, rows):
    for cid in ORDER:
        n, med = _series(rows, cid, 'median'); _, lo = _series(rows, cid, 'q25'); _, hi = _series(rows, cid, 'q75')
        st = S.BBO[cid]
        if cid in ('XY-FMQAOA', 'alpha100', 'alpha1'):
            ax.fill_between(n, lo, hi, color=st['color'], alpha=0.15, lw=0, zorder=1)
        ax.plot(n, med, label=LABEL[cid], zorder=3, **{k:v for k, v in st.items()})
    ax.set_yscale('log'); S.plain_log_axis(ax.yaxis, ticks=[1, 3, 10, 30, 100, 300, 1000])
    ax.set_xlim(40, 100); ax.set_ylim(0.8, 2000)
    ax.set_xlabel('Evaluations'); ax.set_ylabel('True rank of the best solution found\n(1 = global optimum; median over 15 seeds)')


def _success_panel(ax, rows):
    for cid in ORDER:
        n, fr = _series(rows, cid, 'fraction_optimum')
        ax.plot(n, fr, label=LABEL[cid], zorder=3, **S.BBO[cid])
    ax.set_xlim(40, 100); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel('Evaluations'); ax.set_ylabel('Fraction of seeds that found\nthe global optimum')


def fig_bbo():
    rows = read('best_rank_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(S.DCOL_W, 2.7))
    _rank_panel(axes[0], rows); _success_panel(axes[1], rows); S.panel_letters(axes)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=6.5, handlelength=2.6, columnspacing=1.2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=1.2, rect=(0, 0.13, 1, 1)); S.save(fig, HERE, 'bbo_l7_confirm')
    for name, panel in (('best_rank', _rank_panel), ('optimum_fraction', _success_panel)):   # independent panels
        fig, ax = plt.subplots(figsize=(S.COL_W, 3.1)); panel(ax, rows)
        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', ncol=2, fontsize=6.5, handlelength=2.6, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.24, 1, 1)); S.save(fig, HERE/'panels', f'bbo_l7_confirm_{name}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--replot', action='store_true')
    if not ap.parse_args().replot: build_csv()
    fig_bbo(); print('figures written to', HERE)
