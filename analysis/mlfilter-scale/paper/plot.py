"""Paper figures for mlfilter-scale (built on the confirmed closed-loop runs, the exploratory runs and paper-extra).

  python plot.py            rebuild csv/ from the runs (SHA-256 checked), then draw every figure
  python plot.py --replot   draw from csv/ only

Figures (stems): concept, paired_ranks, loop_alpha_sweep, retune_robustness, energy_hist, loop_mechanism,
si_closed_loop_L678; table: csv/compute_cost.csv. Style: src/aps_style.py (no titles).
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
RUNS = PROJECT/'data/runs/mlfilter-scale'
MAIN = ['XY-FMQAOA', 'alpha1', 'alpha100', 'alpha0.3', 'RandomFM', 'Random']
LABEL = {'XY-FMQAOA': r'XY-FMQAOA ($\lambda=0$)', 'alpha1': r'SA, $\alpha=1$', 'alpha100': r'SA, $\alpha=100$',
         'alpha0.3': r'SA, $\alpha=0.3$', 'RandomFM': 'Random proposals + FM selection', 'Random': 'Random search'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(name, rows):
    CSV.mkdir(exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(CSV/name, 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)


def read(name):
    def conv(k, v):
        if v == '': return None
        try: return float(v)
        except ValueError: return v
    with open(CSV/name) as fh:
        return [{k:conv(k, v) for k, v in r.items()} for r in csv.DictReader(fh)]


def ranks_for(truth_run):
    m = json.loads((RUNS/truth_run/'metadata.json').read_text()) if (RUNS/truth_run/'metadata.json').exists() else None
    t = np.load(RUNS/truth_run/'raw/truth_table.npz')['fom']; order = np.argsort(t, kind='stable')
    r = np.empty(len(t), int); r[order] = np.arange(1, len(t)+1); return t, r


def write_input_manifest():
    """SHA-256 of every input file read by build_csv (csv/input_manifest.json)."""
    files = set()
    for key in ('confirm', 'paper_extra', 'classical', 'confirm2'):
        files |= {p for p in (RUNS/CONFIG['runs'][key]).rglob('*') if p.is_file() and p.suffix in ('.json', '.npz') and 'logs' not in p.parts}
    for info in CONFIG['runs']['explore'].values():
        files |= {p for p in (RUNS/info['bbo_xy']/'raw/bbo').glob('*.json')}
        files |= {PROJECT/r for r in json.loads((RUNS/info['bbo_xy']/'metadata.json').read_text())['reused_sha256']}
        files.add(RUNS/info['truth']/'raw/truth_table.npz')
    files.add(RUNS/CONFIG['runs']['truth7']/'raw/truth_table.npz')
    CSV.mkdir(exist_ok=True)
    (CSV/'input_manifest.json').write_text(json.dumps({str(p.relative_to(PROJECT)):sha(p) for p in sorted(files)}, indent=1)+'\n')


# ------------------------------------------------------------------ CSV builders
def _closed_loop_rows(run_file, cid, seed, rank, B=100):
    d = json.loads(Path(run_file).read_text()); ids = d['final_ids'][:B]
    best = np.minimum.accumulate(rank[ids])
    if len(best) < B: best = np.concatenate([best, np.full(B-len(best), best[-1])])
    return d, best


def build_csv():
    write_input_manifest()
    confirm = RUNS/CONFIG['runs']['confirm']; _, rank7 = ranks_for(CONFIG['runs']['truth7'])
    # paired final ranks, mechanism per cycle, compute cost (confirm run)
    finals, mech, cost = [], [], []
    for cid in MAIN:
        for s in CONFIG['confirm_seeds']:
            d, best = _closed_loop_rows(confirm/f'raw/bbo/{cid}_seed{s}.json', cid, s, rank7)
            finals.append({'condition':cid,'seed':s,'best_rank':int(best[-1]),'evaluations_used':len(d['final_ids'])})
            for e in d.get('events', []):
                sel = e['selected_id']
                mech.append({'condition':cid,'seed':s,'cycle':e['cycle'],'selected_true_rank':None if sel is None else int(rank7[sel]),
                             'unseen_fm_min_hit':int(e['unseen_FM_min_hit']),'unique_new_candidates':e['unique_unseen_count'],
                             'feasible_rate':e['raw_feasible_rate']})
                cost.append({'condition':cid,'seed':s,'cycle':e['cycle'],'training_s':e['training_seconds'],'proposal_s':e['sampling_seconds'],
                             'angle_evaluations':e.get('angle_evaluations')})
    write_csv('confirm_final_ranks.csv', finals); write_csv('confirm_mechanism_per_cycle.csv', mech)
    agg = {}
    for r in cost: agg.setdefault(r['condition'], []).append(r)
    write_csv('compute_cost.csv', [{'condition':c,'cycles':len(v),'median_proposal_seconds':float(np.median([x['proposal_s'] for x in v])),
                                    'first_cycle_proposal_seconds_median':float(np.median([x['proposal_s'] for x in v if x['cycle']==1])),
                                    'later_cycles_proposal_seconds_median':float(np.median([x['proposal_s'] for x in v if x['cycle']>1])),
                                    'median_training_seconds':float(np.median([x['training_s'] for x in v])),
                                    'median_angle_evaluations_per_cycle':(float(np.median([x['angle_evaluations'] for x in v if x['cycle']>1]))
                                                                          if v[0]['angle_evaluations'] is not None else None)} for c, v in agg.items()])
    # exploratory closed loop, L = 6, 7, 8 (5 seeds)
    si = []
    for L, info in CONFIG['runs']['explore'].items():
        _, rk = ranks_for(info['truth'])
        files = [(RUNS/info['bbo_xy']/f'raw/bbo/{c}_seed{s}.json', c, s) for c in ('XY-FMQAOA','alpha0.3') for s in CONFIG['explore_seeds']]
        meta = json.loads((RUNS/info['bbo_xy']/'metadata.json').read_text())
        for relp in meta['reused_sha256']:
            d = json.loads((PROJECT/relp).read_text()); files.append((PROJECT/relp, d['condition']['id'], d['seed']))
        for f, c, s in files:
            if c not in MAIN: continue
            _, best = _closed_loop_rows(f, c, s, rk)
            for n in range(40, 101): si.append({'layers':int(L),'condition':c,'seed':s,'evaluations':n,'best_rank':int(best[n-1])})
    write_csv('explore_curves.csv', si)
    # paper-extra: closed-loop alpha sweep + retuned SA, fixed-FM schedules, energy distributions
    pe = RUNS/CONFIG['runs']['paper_extra']
    loop = [{'condition':r['condition'],'seed':r['seed'],'best_rank':r['best_rank'],'source':'confirm'} for r in finals]
    for f in sorted((pe/'raw/loop').glob('*.json')):
        d = json.loads(f.read_text())
        if d['status']!='completed': continue
        _, best = _closed_loop_rows(f, d['condition']['id'], d['seed'], rank7)
        loop.append({'condition':d['condition']['id'],'seed':d['seed'],'best_rank':int(best[-1]),'source':'paper-extra'})
    write_csv('loop_alpha_sweep.csv', loop)
    fixed, hist = [], []
    bins = np.linspace(0, 1, 21)
    for f in sorted((pe/'raw/fixedfm').glob('L*_seed*.json')):
        d = json.loads(f.read_text())
        if d['status']!='completed': continue
        g = {}
        for row in d['batches']: g.setdefault(row['sampler'], []).append(row)
        for smp, rows in g.items():
            v = lambda k: (float(np.mean([r[k] for r in rows if r.get(k) is not None])) if any(r.get(k) is not None for r in rows) else None)
            fixed.append({'layers':d['layers'],'seed':d['seed'],'sampler':smp,'sample_fm_pct':v('sample_fm_pct'),'norm_gap':v('norm_gap'),'feasible':v('feasible')})
        pz = np.load(f.with_name(f.stem+'_pct.npz'))
        for k in pz.files:
            h, _ = np.histogram(pz[k], bins=bins)
            for i in range(len(h)): hist.append({'layers':d['layers'],'seed':d['seed'],'sampler':k,'bin_lo':bins[i],'bin_hi':bins[i+1],'count':int(h[i])})
    write_csv('fixedfm_schedules.csv', fixed); write_csv('energy_hist.csv', hist)
    # classical penalty-free baselines (closed loop L=7 on the 15 confirmation seeds; fixed FM)
    cl = RUNS/CONFIG['runs']['classical']; cls = json.loads((cl/'raw/summary.json').read_text())
    write_csv('classical_loop.csv', [{'condition':r['condition'],'seed':r['seed'],'best_rank':r['best_rank'],'success':int(r['success'])} for r in cls['rows']])
    write_csv('classical_fixed.csv', [{'key':k,**{m:v[m][0] for m in v}} for k, v in cls['fixed'].items()])
    # second confirmation (30 seeds), if finished
    c2 = RUNS/CONFIG['runs']['confirm2']
    if (c2/'raw/summary.json').exists():
        s2 = json.loads((c2/'raw/summary.json').read_text()); fin, cur = [], []
        for key, v in s2['per_seed'].items():
            cid, seed = key.split('|'); fin.append({'condition':cid,'seed':int(seed),'best_rank':v['best_rank'],'success':int(v['success'])})
            _, best = _closed_loop_rows(c2/f'raw/bbo/{cid}_seed{seed}.json', cid, int(seed), rank7)
            for n in range(40, 101): cur.append({'condition':cid,'seed':int(seed),'evaluations':n,'best_rank':int(best[n-1])})
        write_csv('confirm2_final.csv', fin); write_csv('confirm2_curves.csv', cur)
    print('CSV written:', ', '.join(p.name for p in sorted(CSV.glob('*.csv'))))


# ------------------------------------------------------------------ figures
def fig_concept():
    """Schematic. (a)(b): normalised energy (E / largest coefficient) of feasible states (circles) and the infeasible
    states between them (crosses) for a small and a large penalty. (c): the XY mixer only connects feasible states."""
    rng = np.random.default_rng(3); fm = np.sort(rng.normal(0, 1, 7))[::-1][[3, 0, 5, 2, 6, 1, 4]]
    fig, axes = plt.subplots(1, 3, figsize=(S.DCOL_W, 2.2), gridspec_kw={'width_ratios':[1, 1, 0.9]})
    for ax, alpha, name in ((axes[0], 0.3, r'(small penalty, $\alpha=0.3$)'), (axes[1], 30., r'(large penalty, $\alpha=30$)')):
        scale = max(1., alpha)                                   # solver sees energies divided by the largest coefficient
        xf = np.arange(len(fm))*2; xi = xf[:-1] + 1
        ef = fm/scale; ei = (0.5*(fm[:-1]+fm[1:]) - 2.0 + alpha*1.0)/scale   # one violated constraint between neighbours
        xs = np.concatenate([[xf[0]], np.ravel(np.column_stack([xi, xf[1:]]))]); es = np.concatenate([[ef[0]], np.ravel(np.column_stack([ei, ef[1:]]))])
        ax.plot(xs, es, color=S.MUTED, lw=0.8, zorder=1)
        ax.plot(xf, ef, 'o', color=S.BBO['alpha1']['color'], ms=4, label='feasible', zorder=3)
        ax.plot(xi, ei, 'x', color=S.BBO['alpha100']['color'], ms=4, mew=1, label='infeasible', zorder=3)
        ax.set_xticks([]); ax.set_ylabel('Normalised energy'); ax.set_xlabel('States along a path\n' + name)
        ax.minorticks_off()
    axes[0].legend(loc='upper right', fontsize=6.5)
    ax = axes[2]; n = 4; th = np.linspace(0, 2*np.pi, n, endpoint=False) + np.pi/4
    for k, cx in enumerate((0.0, 2.6)):
        x, y = cx + np.cos(th), np.sin(th)
        for i in range(n):
            j = (i+1) % n; ax.annotate('', xy=(x[j], y[j]), xytext=(x[i], y[i]), arrowprops=dict(arrowstyle='<->', color=S.BBO['XY-FMQAOA']['color'], lw=1.0, shrinkA=6, shrinkB=6))
        for i in range(n):
            ax.plot(x[i], y[i], 'o', ms=11, mfc='white', mec=S.INK, mew=0.8)
            ax.text(x[i], y[i], ''.join('1' if t == i else '0' for t in range(n)), ha='center', va='center', fontsize=5)
        ax.text(cx, -1.55, f'layer {k+1}', ha='center', fontsize=7)
    ax.text(1.3, 1.45, r'XY mixer: $|1000\rangle\leftrightarrow|0100\rangle\leftrightarrow\cdots$', ha='center', fontsize=6.5)
    ax.set_xlim(-1.4, 4.0); ax.set_ylim(-1.8, 1.7); ax.axis('off')
    S.panel_letters(axes); fig.tight_layout(w_pad=0.8); S.save(fig, HERE, 'concept')


def fig_paired():
    """Per-seed final best rank (log): XY vs each comparator, lines join the same seed."""
    rows = read('confirm_final_ranks.csv'); seeds = sorted({r['seed'] for r in rows}); get = {(r['condition'], r['seed']):r['best_rank'] for r in rows}
    comps = ['alpha0.3', 'alpha100', 'alpha1']
    fig, axes = plt.subplots(1, 3, figsize=(S.DCOL_W, 2.4), sharey=True)
    for ax, c in zip(axes, comps):
        for s in seeds:
            a, b = get[(c, s)], get[('XY-FMQAOA', s)]
            col = S.BBO['XY-FMQAOA']['color'] if b < a else (S.MUTED if b == a else S.BBO['alpha100']['color'])
            ax.plot([0, 1], [a, b], '-', color=col, lw=0.8, alpha=0.9); ax.plot([0, 1], [a, b], 'o', color=col, ms=3)
        w = sum(get[('XY-FMQAOA', s)] < get[(c, s)] for s in seeds); t = sum(get[('XY-FMQAOA', s)] == get[(c, s)] for s in seeds)
        ax.text(0.5, 0.97, f'XY better {w}, tie {t}, worse {len(seeds)-w-t}', transform=ax.transAxes, ha='center', va='top', fontsize=6.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels([LABEL[c], 'XY-FMQAOA'], fontsize=7); ax.set_xlim(-0.3, 1.3); ax.minorticks_off()
        ax.set_yscale('log'); S.plain_log_axis(ax.yaxis, ticks=[1, 3, 10, 30, 100, 300])
    axes[0].set_ylabel('True rank of the best solution\nafter 100 evaluations')
    S.panel_letters(axes); fig.tight_layout(w_pad=0.6); S.save(fig, HERE, 'paired_ranks')


def fig_loop_mechanism():
    """Per cycle (median over 15 seeds, 5-cycle running mean): (a) true rank of the selected candidate,
    (b) fraction of cycles that selected the unseen FM minimum, (c) number of distinct new feasible candidates offered."""
    rows = read('confirm_mechanism_per_cycle.csv'); conds = ['XY-FMQAOA', 'alpha1', 'alpha100', 'alpha0.3', 'RandomFM']
    fig, axes = plt.subplots(1, 3, figsize=(S.DCOL_W, 2.5))
    def series(c, key, fn):
        cyc = sorted({r['cycle'] for r in rows if r['condition']==c}); v = []
        for k in cyc:
            x = [r[key] for r in rows if r['condition']==c and r['cycle']==k and r[key] is not None]
            v.append(fn(x) if x else np.nan)
        v = np.array(v, float); w = 5; sm = np.array([np.nanmean(v[max(0, i-w+1):i+1]) for i in range(len(v))]); return np.array(cyc), sm
    for c in conds:
        st = S.BBO[c]
        x, y = series(c, 'selected_true_rank', np.median); axes[0].plot(x, y, label=LABEL[c], **st)
        x, y = series(c, 'unseen_fm_min_hit', np.mean); axes[1].plot(x, y, **st)
        x, y = series(c, 'unique_new_candidates', np.median); axes[2].plot(x, y, **st)
    axes[0].set_yscale('log'); S.plain_log_axis(axes[0].yaxis, ticks=[10, 30, 100, 300, 1000, 3000])
    axes[0].set_ylabel('True rank of the\nselected candidate'); axes[1].set_ylabel('Fraction selecting the\nunseen FM minimum'); axes[2].set_ylabel('Distinct new feasible\ncandidates per cycle')
    axes[1].set_ylim(-0.03, 1.03)
    for ax in axes: ax.set_xlabel('Proposal cycle'); ax.set_xlim(1, 60)
    S.panel_letters(axes); h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=6.5, handlelength=2.6, columnspacing=1.0, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=0.6, rect=(0, 0.1, 1, 1)); S.save(fig, HERE, 'loop_mechanism')


def fig_si_explore():
    """Exploratory closed loop (5 seeds): median best rank vs evaluations for L = 6, 7, 8."""
    rows = read('explore_curves.csv')
    fig, axes = plt.subplots(1, 3, figsize=(S.DCOL_W, 2.6), sharey=True)
    for ax, L in zip(axes, (6, 7, 8)):
        for c in MAIN:
            r = [x for x in rows if x['layers']==L and x['condition']==c]
            if not r: continue
            n = np.arange(40, 101); med = [np.median([x['best_rank'] for x in r if x['evaluations']==k]) for k in n]
            ax.plot(n, med, label=LABEL[c], **S.BBO[c])
        ax.set_yscale('log'); S.plain_log_axis(ax.yaxis, ticks=[1, 3, 10, 30, 100, 300, 1000]); ax.set_xlim(40, 100)
        ax.set_xlabel('Evaluations'); ax.text(0.97, 0.95, f'$L={L}$', transform=ax.transAxes, ha='right', va='top', fontsize=7)
    axes[0].set_ylabel('True rank of the best solution\n(median over 5 seeds)')
    S.panel_letters(axes); h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=6.5, handlelength=2.6, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=0.6, rect=(0, 0.14, 1, 1)); S.save(fig, HERE, 'si_closed_loop_L678')


def fig_loop_alpha_sweep():
    """Closed loop, L=7, 15 seeds: final best rank vs alpha (SA, automatic schedule) with XY as a horizontal band."""
    rows = read('loop_alpha_sweep.csv'); alphas = [0.3, 1, 3, 10, 30, 100, 1000]
    def stat(c):
        v = [r['best_rank'] for r in rows if r['condition']==c]; return (np.median(v), np.percentile(v, 25), np.percentile(v, 75), len(v)) if v else None
    fig, ax = plt.subplots(figsize=(S.COL_W, 2.6))
    sa = [(a, stat(f'alpha{a:g}')) for a in alphas]; sa = [(a, s) for a, s in sa if s]
    a = np.array([x for x, _ in sa]); m = np.array([s[0] for _, s in sa]); lo = np.array([s[1] for _, s in sa]); hi = np.array([s[2] for _, s in sa])
    ax.fill_between(a, lo, hi, color=S.BAND, lw=0, zorder=0); ax.plot(a, m, label=r'SA with penalty $\alpha$ (median, IQR)', zorder=3, **S.SA)
    xy = stat('XY-FMQAOA'); ax.axhspan(xy[1], xy[2], color=S.BBO['XY-FMQAOA']['color'], alpha=0.15, lw=0)
    ax.axhline(xy[0], label=r'XY-FMQAOA ($\lambda=0$)', **S.BBO['XY-FMQAOA'])
    ad = stat('Adaptive')
    if ad: ax.axhline(ad[0], label=r'SA, adaptive $\alpha$', color=S.INK, ls=':', lw=1.0)
    ax.axhline(stat('RandomFM')[0], label='Random proposals + FM selection', **S.RANDOMFM)
    ax.set_xscale('log'); S.plain_log_axis(ax.xaxis, ticks=alphas); ax.set_xlim(0.2, 1600)
    ax.set_yscale('log'); S.plain_log_axis(ax.yaxis, ticks=[1, 3, 10, 30, 100]); ax.set_ylim(0.8, 300)
    ax.set_xlabel(r'Penalty coefficient $\alpha$ ($\lambda=\alpha S$)'); ax.set_ylabel('True rank of the best solution\nafter 100 evaluations')
    opt = lambda c: sum(1 for r in rows if r['condition']==c and r['best_rank']==1)
    ax.text(0.98, 0.04, f"optimum found: XY {opt('XY-FMQAOA')}/15, adaptive {opt('Adaptive')}/15, " + r'$\alpha$=1 ' + f"{opt('alpha1')}/15",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6)
    ax.legend(loc='upper center', fontsize=6, ncol=1); fig.tight_layout(); S.save(fig, HERE, 'loop_alpha_sweep')


SCHEDULE_STYLE = {'auto': dict(color=S.INK, ls='-', marker='o'), 'fmscale': dict(color='#eb6834', ls='--', marker='s', mfc='white'),
                  'wide10x': dict(color='#1baf7a', ls='-.', marker='^', mfc='white'), 'fmscale10x': dict(color='#2a78d6', ls=':', marker='D', mfc='white')}
SCHEDULE_LABEL = {'auto': 'automatic (1000 sweeps)', 'fmscale': 'FM-scaled range', 'wide10x': 'automatic, 10x sweeps', 'fmscale10x': 'FM-scaled, 10x sweeps'}


def fig_retune():
    """Fixed FM: sample FM percentile vs alpha for SA temperature schedules (L = 6, 7, 8; median over 5 FMs);
    inset row: closed-loop alpha=100 with retuned schedules (L=7, 15 seeds)."""
    rows = read('fixedfm_schedules.csv'); alphas = [1, 3, 10, 30, 100, 1000]
    fig, axes = plt.subplots(1, 4, figsize=(S.DCOL_W, 2.6), gridspec_kw={'width_ratios':[1, 1, 1, 0.9]})
    for ax, L in zip(axes[:3], (6, 7, 8)):
        for sch, st in SCHEDULE_STYLE.items():
            m = [np.median([r['sample_fm_pct'] for r in rows if r['layers']==L and r['sampler']==f'SA-{sch}-a{a:g}' and r['sample_fm_pct'] is not None] or [np.nan]) for a in alphas]
            ax.plot(alphas, m, label=SCHEDULE_LABEL[sch], ms=3.5, lw=1.0, **st)
        rf = np.median([r['sample_fm_pct'] for r in rows if r['layers']==L and r['sampler']=='RandomFM'])
        xy = np.median([r['sample_fm_pct'] for r in rows if r['layers']==L and r['sampler']=='XY-p4-cvar10'])
        ax.axhline(rf, **S.RANDOMFM, label='Random proposals'); ax.axhline(xy, label=r'XY-FMQAOA $p=4$', **S.BBO['XY-FMQAOA'])
        ax.set_xscale('log'); S.plain_log_axis(ax.xaxis, ticks=alphas); ax.set_ylim(-0.02, 0.56)
        ax.set_xlabel(r'$\alpha$'); ax.text(0.03, 0.95, f'$L={L}$', transform=ax.transAxes, ha='left', va='top', fontsize=7)
    axes[0].set_ylabel('Mean FM percentile of samples\n(0 = FM best, 0.5 = random)')
    loop = read('loop_alpha_sweep.csv'); conds = [('alpha100', 'auto'), ('alpha100-fmscale', 'fmscale'), ('alpha100-wide10x', 'wide10x'),
                                                  ('alpha100-fmscale10x', 'fmscale10x'), ('alpha1', None), ('XY-FMQAOA', None)]
    ax = axes[3]
    for i, (c, sch) in enumerate(conds):
        v = [r['best_rank'] for r in loop if r['condition']==c]
        if not v: continue
        col = SCHEDULE_STYLE[sch]['color'] if sch else (S.BBO[c]['color'])
        ax.plot([i]*len(v) + 0.12*np.random.default_rng(i).uniform(-1, 1, len(v)), v, 'o', ms=2.5, color=col, alpha=0.6)
        ax.plot([i-0.3, i+0.3], [np.median(v)]*2, color=col, lw=1.6)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([r'$\alpha$=100' + '\n' + 'auto', 'FM-\nscaled', 'auto\n10x', 'FM-sc.\n10x', r'$\alpha$=1', 'XY'], fontsize=5.5)
    ax.minorticks_off(); ax.set_yscale('log'); S.plain_log_axis(ax.yaxis, ticks=[1, 3, 10, 30, 100]); ax.set_ylabel('Closed-loop best rank (L=7)')
    S.panel_letters(axes); h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=6.5, handlelength=2.6, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=0.5, rect=(0, 0.15, 1, 1)); S.save(fig, HERE, 'retune_robustness')


def fig_energy_hist():
    """Fixed FM, L=7: distribution of the FM percentile of feasible samples (pooled over 5 FMs x 10 reps, density)."""
    rows = read('energy_hist.csv'); L = 7
    samplers = [('SA-auto-a1', r'SA, $\alpha=1$', dict(color=S.INK, ls='-')), ('SA-auto-a10', r'SA, $\alpha=10$', dict(color='#eb6834', ls='--')),
                ('SA-auto-a100', r'SA, $\alpha=100$', dict(color='#1baf7a', ls='-.')), ('XY-p4-cvar10', r'XY-FMQAOA $p=4$', dict(color=S.BBO['XY-FMQAOA']['color'], ls='-', lw=1.6)),
                ('RandomFM', 'Uniform random', dict(color=S.MUTED, ls=':'))]
    fig, ax = plt.subplots(figsize=(S.COL_W, 2.5))
    for key, lab, st in samplers:
        r = [x for x in rows if x['layers']==L and x['sampler']==key]
        if not r: continue
        lo = sorted({x['bin_lo'] for x in r}); cnt = np.array([sum(x['count'] for x in r if x['bin_lo']==b) for b in lo], float)
        dens = cnt/cnt.sum()/(lo[1]-lo[0]); edges = np.append(lo, 1.0)
        ax.stairs(dens, edges, label=lab, lw=st.pop('lw', 1.1), **st)
    ax.set_yscale('log'); ax.set_xlim(0, 1); ax.set_xlabel('FM percentile of a feasible sample (0 = FM best)'); ax.set_ylabel('Density')
    ax.legend(loc='upper right', fontsize=6.5); fig.tight_layout(); S.save(fig, HERE, 'energy_hist')


SHAPE = {'XY-FMQAOA': dict(color='#2a78d6', ls='-', lw=1.6), 'Boltzmann-t0.2': dict(color='#1baf7a', ls='-.', lw=1.2),
         'CatSA': dict(color='#eb6834', ls='--', lw=1.2), 'alpha1': dict(color=S.INK, ls='--', lw=1.0), 'alpha100': dict(color=S.MUTED, ls='-', lw=1.0),
         'alpha0.3': dict(color=S.MUTED, ls=':', lw=1.2), 'RandomFM': dict(color=S.MUTED, ls='--', lw=1.0)}
SHAPE_LABEL = {'XY-FMQAOA': r'XY-FMQAOA ($\lambda=0$)', 'Boltzmann-t0.2': r'Boltzmann, $\tau=0.2$', 'CatSA': 'Categorical SA',
               'alpha1': r'SA, $\alpha=1$', 'alpha100': r'SA, $\alpha=100$', 'alpha0.3': r'SA, $\alpha=0.3$', 'RandomFM': 'Random proposals + FM selection'}


def fig_shape():
    """Closed loop (L=7, 15 confirmation seeds): fraction of seeds that found the global optimum within 100 evaluations
    for the Boltzmann proposal vs temperature tau; XY-FMQAOA, categorical SA and tuned SA as horizontal references."""
    rows = read('classical_loop.csv'); taus = [0.02, 0.05, 0.1, 0.2, 0.5]
    frac = lambda c: float(np.mean([r['success'] for r in rows if r['condition']==c]))
    fig, ax = plt.subplots(figsize=(S.COL_W, 2.6))
    ax.plot(taus, [frac(f'Boltzmann-t{t:g}') for t in taus], 'o-', color='#1baf7a', label=r'Boltzmann proposal $\propto e^{-\mathrm{FM}/T}$, $T=\tau\,\sigma_\mathrm{FM}$', zorder=3)
    for c in ('XY-FMQAOA', 'alpha1', 'CatSA'):
        ax.axhline(frac(c), label=SHAPE_LABEL[c], zorder=2, **SHAPE[c])
    ax.set_xscale('log'); S.plain_log_axis(ax.xaxis, ticks=taus); ax.set_xlim(0.015, 0.65); ax.set_ylim(0, 1.0)
    ax.set_xlabel(r'Proposal temperature $\tau$ (wider $\rightarrow$)'); ax.set_ylabel('Fraction of seeds that found the\nglobal optimum (100 evaluations)')
    ax.legend(loc='lower right', fontsize=6); fig.tight_layout(); S.save(fig, HERE, 'shape_tau_sweep')


def fig_confirm2():
    """Second confirmation (L=7, 30 new seeds): (a) median best rank vs evaluations, (b) fraction of seeds at the optimum."""
    rows = read('confirm2_curves.csv'); conds = ['XY-FMQAOA', 'Boltzmann-t0.2', 'CatSA', 'alpha1', 'alpha100', 'alpha0.3']
    fig, axes = plt.subplots(1, 2, figsize=(S.DCOL_W, 2.6))
    for c in conds:
        r = [x for x in rows if x['condition']==c]; n = np.arange(40, 101)
        axes[0].plot(n, [np.median([x['best_rank'] for x in r if x['evaluations']==k]) for k in n], label=SHAPE_LABEL[c], **SHAPE[c])
        axes[1].plot(n, [np.mean([x['best_rank']==1 for x in r if x['evaluations']==k]) for k in n], **SHAPE[c])
    axes[0].set_yscale('log'); S.plain_log_axis(axes[0].yaxis, ticks=[1, 3, 10, 30, 100, 300]); axes[0].set_ylabel('True rank of the best solution\n(median over 30 seeds)')
    axes[1].set_ylim(-0.03, 1.03); axes[1].set_ylabel('Fraction of seeds that found\nthe global optimum')
    for ax in axes: ax.set_xlim(40, 100); ax.set_xlabel('Evaluations')
    S.panel_letters(axes); h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=6.5, handlelength=2.6, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=1.2, rect=(0, 0.14, 1, 1)); S.save(fig, HERE, 'confirm2')


def draw_all():
    fig_concept(); fig_paired(); fig_loop_mechanism(); fig_si_explore()
    if any(r['source']=='paper-extra' for r in read('loop_alpha_sweep.csv')): fig_loop_alpha_sweep()
    if (CSV/'fixedfm_schedules.csv').exists() and read('fixedfm_schedules.csv'): fig_retune(); fig_energy_hist()
    if (CSV/'classical_loop.csv').exists(): fig_shape()
    if (CSV/'confirm2_curves.csv').exists(): fig_confirm2()
    print('figures written to', HERE)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--replot', action='store_true')
    if not ap.parse_args().replot: build_csv()
    draw_all()
