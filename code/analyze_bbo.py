"""Audit a complete run before exporting standalone scientific figures and reports."""
import argparse
import json
import itertools
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from config import load_config, parse_custom_xy_qaoa, parse_custom_standard_qaoa
from evaluator import load_problem
from main import _save_json, export_csv_tables, summarize_results
from shared_numerics import AdaptivePenaltyTracker

STYLES={
 'Exact-FM':('#333333','x',':'),'Adaptive-FMQA':('#1f77b4','o','-'),
 'LargePenalty-FMQA':('#ff7f0e','s','--'),'Penalty-FMQAOA':('#d62728','^','-.'),
 'XY-FMQAOA':('#2ca02c','D','-')}

for depth in [2,3]:
    for family,color in [('Penalty-FMQAOA','#d62728'),('XY-FMQAOA','#2ca02c')]:
        STYLES[f'{family}(p={depth},mean,powell49)']=(color,'o' if depth==2 else 's','--' if depth==2 else ':')


def short_name(method):
    if 'QAOA' not in method:
        return {'Exact-FM':'Exact FM','Adaptive-FMQA':'Adaptive SA','LargePenalty-FMQA':'Large SA'}[method]
    return ('XY' if method.startswith('XY') else 'X') + ' p=' + (method.split('p=')[1].split(',')[0] if '(p=' in method else '1')



def audit(records,cfg,p):
    assert len(records)==len(cfg['seeds'])
    for rec,seed in zip(records,cfg['seeds']):
        assert rec['seed']==seed and rec['dataset_sha256']==p.data_sha256
        initial=p.initial_dataset(seed,cfg['problem']['initial_samples'])
        assert rec['initial_candidate_ids']==initial
        assert set(rec['methods'])==set(cfg['methods'])
        assert len({r['step0_fm_sha256'] for r in rec['methods'].values()})==1
        for method in cfg['methods']:
            r=rec['methods'][method];assert r['status']=='completed'
            assert len(r['history'])==cfg['problem']['bbo_cycles']
            seen=set(initial);best=min(p.evaluate_id(i) for i in initial)
            tracker=AdaptivePenaltyTracker(initial_alpha=cfg['penalty'].get('adaptive_initial_alpha',1.))
            for step,h in enumerate(r['history'],1):
                assert h['step']==step
                cid=h['chosen_candidate_id']
                if cid is None:assert h['chosen_true_objective'] is None
                else:
                    assert cid not in seen and 0<=cid<len(p.categories)
                    np.testing.assert_allclose(h['chosen_true_objective'],p.evaluate_id(cid),atol=0,rtol=0)
                    seen.add(cid);best=min(best,p.evaluate_id(cid))
                assert h['total_bb_evaluations']==len(seen)
                np.testing.assert_allclose(h['best_objective_so_far'],best,atol=1e-12,rtol=0)
                np.testing.assert_allclose(h['normalized_regret_so_far'],p.normalized_regret(best),atol=1e-12,rtol=0)
                sm=h['sampler']
                alpha=sm['penalty_alpha'];lam=sm['penalty_lambda'];S=h['ising_scale_S']
                np.testing.assert_allclose(lam,S*alpha,atol=1e-10,rtol=1e-10)
                if method=='Adaptive-FMQA':
                    np.testing.assert_allclose(alpha,tracker.get_current_alpha(),atol=0,rtol=0)
                    np.testing.assert_allclose(h['next_alpha'],tracker.update(h['raw_feasible_rate']),atol=0,rtol=0)
                if method=='LargePenalty-FMQA':assert alpha==100
                if method.startswith('Penalty-FMQAOA'):assert alpha==5
                if 'QAOA' in method:
                    expected_depth=int(method.split('p=')[1].split(',')[0]) if '(p=' in method else 1
                    assert sm['depth']==expected_depth and sm['objective']=='mean'
                    assert sm['angle_search_mode']=='powell49'
                    assert sm['angle_evaluations']==49 and sm['shots']==100
                if method.startswith('XY-FMQAOA'):
                    assert lam==0 and h['invalid_count']==0
                    assert abs(1-sm['statevector_feasible_prob'])<1e-10
                if method=='Exact-FM':assert abs(h['selected_fm_gap'])<1e-8
            assert r['total_bb_evaluations']==len(seen)<=cfg['problem']['initial_samples']+cfg['problem']['bbo_cycles']
            np.testing.assert_allclose(r['final_best_objective'],best,atol=1e-12,rtol=0)


def render(records,cfg,out):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    methods=cfg['methods'];steps=np.arange(cfg['problem']['bbo_cycles']+1);n_initial=cfg['problem']['initial_samples']
    def save(fig,name):
        fig.tight_layout()
        for fmt in ['pdf','svg','png']:
            (out/fmt).mkdir(exist_ok=True);fig.savefig(out/fmt/f'{name}.{fmt}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for m in methods:
        vals=np.array([[r['initial_normalized_regret']]+[h['normalized_regret_so_far'] for h in r['methods'][m]['history']] for r in records])
        lo,med,hi=np.percentile(vals,[25,50,75],axis=0);c,marker,ls=STYLES[m]
        ax.plot(steps,med,color=c,marker=marker,markevery=max(1,len(steps)//5),ls=ls,label=m)
        ax.fill_between(steps,lo,hi,color=c,alpha=.12)
    ax.set(xlabel=f'Proposal attempts after {n_initial} initial evaluations',ylabel='Normalized regret')
    ax.legend(frameon=False,fontsize=7,ncol=2);ax.grid(alpha=.15);save(fig,'regret_vs_proposals')
    fig,ax=plt.subplots(figsize=(8,5))
    for m in methods:
        c,marker,ls=STYLES[m]
        for i,r in enumerate(records):
            hist=r['methods'][m]['history']
            ax.plot([r['initial_samples']]+[h['total_bb_evaluations'] for h in hist],
                    [r['initial_best_objective']]+[h['best_objective_so_far'] for h in hist],
                    color=c,ls=ls,alpha=.55,lw=1.1,label=m if i==0 else None)
    ax.set(xlabel=f'Actual objective lookups (including initial {n_initial})',ylabel='Best optical FOM')
    ax.legend(frameon=False,fontsize=7,ncol=2);ax.grid(alpha=.15);save(fig,'best_vs_actual_evaluations')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    short=[short_name(m) for m in methods]
    for ax,key,ylabel in zip(axes,['mean_raw_feasible_rate','total_bb_evaluations'],['Raw feasible fraction','Actual objective lookups']):
        for i,m in enumerate(methods):
            v=np.array([r['methods'][m][key] for r in records]);lo,med,hi=np.percentile(v,[25,50,75]);c,marker,_=STYLES[m]
            ax.errorbar(i,med,yerr=[[med-lo],[hi-med]],fmt=marker,color=c,capsize=4)
            ax.scatter(i+np.linspace(-.12,.12,len(v)),v,color=c,s=12,alpha=.5)
        ax.set_xticks(range(len(methods)),short,rotation=30,ha='right');ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylim(-.02,1.05);save(fig,'feasibility_and_evaluation_counts')
    fig,ax=plt.subplots(figsize=(8,4.5))
    for i,m in enumerate([x for x in methods if x!='XY-FMQAOA']):
        v=np.array([r['methods']['XY-FMQAOA']['final_normalized_regret']-r['methods'][m]['final_normalized_regret'] for r in records])
        lo,med,hi=np.percentile(v,[25,50,75]);c,_,_=STYLES[m]
        ax.scatter(i+np.linspace(-.12,.12,len(v)),v,color=c,alpha=.6)
        ax.errorbar(i,med,yerr=[[med-lo],[hi-med]],fmt='D',color='black',capsize=5)
    ax.axhline(0,color='#777777',lw=1);ax.set_xticks(range(len(methods)-1),[short_name(m) for m in methods if m!='XY-FMQAOA'],rotation=30,ha='right')
    ax.set_ylabel('Paired final regret difference (XY - comparator)');save(fig,'paired_regret_differences')
    fig,ax=plt.subplots(figsize=(8,4.5))
    for r in records:
        ax.plot(steps[1:],[h['sampler']['penalty_alpha'] for h in r['methods']['Adaptive-FMQA']['history']],label=f"Seed {r['seed']}")
    ax.set(xlabel='Proposal attempt',ylabel='Adaptive normalized penalty alpha',yscale='log');ax.legend(frameon=False);save(fig,'adaptive_penalty')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=load_config(['-c',str(args.config)]);out=(Path(__file__).resolve().parent/cfg['output_dir']).resolve()
    p=load_problem(cfg['problem']);records=[json.loads((out/f'seeds/seed{s}/json/seed_{s}.json').read_text()) for s in cfg['seeds']]
    protocol=json.loads((out/'json/protocol.json').read_text())
    for rec in records:
        assert all(r['step0_fm_sha256']==protocol['step0_model_sha256'][str(rec['seed'])] for r in rec['methods'].values())
        local_cfg=load_config(['-c',str(out/f"json/run_config_seed{rec['seed']}.json")])
        assert rec['config_sha256']==hashlib.sha256(json.dumps(local_cfg,sort_keys=True).encode()).hexdigest()
    audit(records,cfg,p)
    summary=summarize_results(records,cfg['methods'])
    paired=[];pairs=list(itertools.combinations(cfg['methods'],2))
    for a,b in pairs:
        delta=np.array([r['methods'][a]['final_normalized_regret']-r['methods'][b]['final_normalized_regret'] for r in records])
        p_raw=1. if np.all(abs(delta)<1e-12) else float(wilcoxon(delta,zero_method='wilcox',alternative='two-sided').pvalue)
        lo,med,hi=np.percentile(delta,[25,50,75])
        paired.append({'pair':f'{a} vs {b}','metric':'final_normalized_regret','per_seed_differences':dict(zip(map(str,cfg['seeds']),delta.tolist())),
                       'diff_median_iqr':{'median':med,'q25':lo,'q75':hi},'p_value_raw':p_raw,'p_value_bonferroni':min(1.,p_raw*len(pairs))})
    summary['paired_tests']=paired
    _save_json(out/'json/summary.json',summary)
    for rec in records:_save_json(out/f"json/seed_{rec['seed']}.json",rec)
    export_csv_tables(records,out/'csv');render(records,cfg,out)
    _save_json(out/'json/audit.json',{'status':'passed','seeds':cfg['seeds'],'methods':cfg['methods'],
        'runs':len(records)*len(cfg['methods']),'proposals_per_run':cfg['problem']['bbo_cycles'],
        'figures':'generated;manual visual review pending','statistics':'median/IQR,paired signed-rank,Bonferroni all36 comparisons'})
    md=out/'md';md.mkdir(exist_ok=True)
    rows=['# Results','',f"Audited {len(records)} seeds × {len(cfg['methods'])} methods. Exploratory pilot.",'',
          '| Method | Final FOM median [Q25,Q75] | Final regret median [Q25,Q75] | Actual evaluations per seed |',
          '|---|---|---|---|']
    for m in cfg['methods']:
        a=summary['methods'][m]['final_best_objective'];b=summary['methods'][m]['final_normalized_regret']
        counts=[r['methods'][m]['total_bb_evaluations'] for r in records]
        rows.append(f"| {m} | {a['median']:.6g} [{a['q25']:.6g},{a['q75']:.6g}] | {b['median']:.6g} [{b['q25']:.6g},{b['q75']:.6g}] | {counts} |")
    rows += ['','Paired final-regret tests:','', '```json',json.dumps(summary['paired_tests'],indent=2),'```','',
             'Figure captions: regret_vs_proposals uses median and 25–75th percentiles across all five seeds; proposal attempts include candidate failures. best_vs_actual_evaluations shows every seed separately without extrapolation. Feasibility/counts show all seeds plus median/IQR. Paired differences use the same seeds, negative favors XY. Adaptive penalty shows every seed. All axes are linear except the logarithmic alpha axis. No figure titles or bottom annotations.','',
             'Source: charged lookups from pinned optical simulations, not new TMM. Figures require manual visual review.']
    (md/'RESULTS.md').write_text('\n'.join(rows)+'\n')
    (md/'DISCUSSION.md').write_text('# Interpretation limits\n\nFive-seed exploratory comparison; no claim of convergence or quantum advantage. Exact-FM minimizes the surrogate, not the true optical objective. Proposal failures and all seeds are retained. Compare actual evaluation counts before interpreting sample efficiency. Equal angle calls/shots do not imply equal runtime or circuit cost. Timing excludes fresh optical simulations. Manual figure review remains pending.\n')
    print('Audit, statistics, reports and five figure sets completed; visual review pending.',flush=True)


if __name__=='__main__':main()
