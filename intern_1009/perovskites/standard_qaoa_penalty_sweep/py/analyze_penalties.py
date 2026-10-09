"""Verify and report Standard-QAOA penalty/shot sensitivity."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from run_penalties import OUT,PILOT,sha,save,PerovskitesEvaluator

KEYS=['empirical_raw_feasible_rate','unseen_success','fm_minimum_hit','unique_unseen_candidates',
      'selected_fm_gap','selected_true_gap','normalized_regret_initial_plus_proposals','feasible_rate_abs_error']

def stats(vals):
    vals=[v for v in vals if v is not None]
    return {'median':float(np.median(vals)) if vals else None,'q1':float(np.quantile(vals,.25)) if vals else None,
      'q3':float(np.quantile(vals,.75)) if vals else None,'defined_seeds':len(vals)}

def main():
    protocol=json.loads((OUT/'json/protocol.json').read_text());rows=[];exact=[];verified=0
    bb=PerovskitesEvaluator();cats=[list(c) for c in bb.candidates()];patterns=np.array([bb.encode(c) for c in cats],dtype=np.int8)
    lookup={tuple(p):i for i,p in enumerate(patterns)}
    for seed in protocol['seeds']:
        d=json.loads((OUT/f'json/seed_{seed}.json').read_text());base=json.loads((PILOT/f'json/seed_{seed}.json').read_text())
        assert d['status']=='completed' and d['protocol_sha256']==sha(OUT/'json/protocol.json')
        assert d['source_sha256']==sha(PILOT/f'json/seed_{seed}.json') and d['checkpoint_sha256']==sha(PILOT/base['checkpoint'])
        assert d['initial_sha256']==base['initial_sha256'] and d['fixed_model_sha256']==base['fixed_model_sha256']
        assert [c['alpha'] for c in d['conditions']]==protocol['alphas']
        pred=np.array(base['fm_quality']['all_candidate_predictions']);initial=set(base['initial_dataset']['candidate_ids'])
        for c in d['conditions']:
            assert len(c['angle_search'])==9 and abs(c['lambda_internal']-c['alpha']*c['base_scale'])<1e-9
            assert abs(c['raw_norm']-1)<1e-10
            best=min(c['angle_search'],key=lambda a:a['expected_normalized_surrogate_plus_penalty'])
            assert c['angles']==[best['gamma'],best['beta']]
            assert list(c['shots'])==[str(n) for n in protocol['shot_counts']]
            exact.append({'seed':seed,'alpha':c['alpha'],'angles':c['angles'],**{k:c[k] for k in [
              'exact_feasible_probability','exact_unseen_probability','exact_fm_minimum_probability',
              'exact_unseen_FM_top5pct_probability','exact_unseen_true_top5pct_probability','exact_energy_mean','exact_energy_variance',
              'base_FM_mean','base_FM_variance','penalized_normalized_Ising_scale']}})
            for ntext,runs in c['shots'].items():
                n=int(ntext);assert len(runs)==100
                for rep,r in enumerate(runs):
                    assert r['repetition']==rep
                    indices=np.array(r['raw_basis_indices'],dtype=np.uint64);assert len(indices)==n
                    bits=((indices[:,None]>>np.arange(23,dtype=np.uint64))&1).astype(np.int8)
                    ids=[lookup.get(tuple(b)) for b in bits];available={i for i in ids if i is not None}-initial
                    expected=min(available,key=lambda i:(pred[i],tuple(patterns[i]))) if available else None
                    assert r['accepted_count']==int(expected is not None) and r['allowed_total_bb_budget']==21
                    if expected is not None:
                        assert r['accepted_candidates'][0]['candidate_id']==expected
                        assert r['selected_true_gap']==bb.evaluate(cats[expected])
                    assert abs(r['empirical_raw_feasible_rate']-sum(i is not None for i in ids)/n)<1e-12
                    verified+=1
                row={'seed':seed,'alpha':c['alpha'],'shots':n,'metrics':{},'defined_repetitions':{}}
                for k in KEYS:
                    vals=[r[k] for r in runs if r[k] is not None];row['metrics'][k]=float(np.mean(vals)) if vals else None
                    row['defined_repetitions'][k]=len(vals)
                errors=np.array([r['energy_error'] for r in runs]);base_errors=np.array([r['base_FM_energy_error'] for r in runs])
                rmse=float(np.sqrt(np.mean(errors**2)))
                row['metrics'].update({'energy_rmse':rmse,'energy_bias':float(errors.mean()),
                  'base_FM_energy_rmse':float(np.sqrt(np.mean(base_errors**2))),
                  'coefficient_scaled_energy_rmse':rmse/c['penalized_normalized_Ising_scale'],
                  'theoretical_energy_se':float(np.sqrt(c['exact_energy_variance']/n)),
                  'exact_unseen_batch_success':float(-np.expm1(n*np.log1p(-min(c['exact_unseen_probability'],1-1e-15))))})
                rows.append(row)
    keys=KEYS+['energy_rmse','energy_bias','base_FM_energy_rmse','coefficient_scaled_energy_rmse','theoretical_energy_se','exact_unseen_batch_success']
    summary={'status':'completed','verified_conditions':verified,'protocol_sha256':sha(OUT/'json/protocol.json'),
      'seed_level_repetition_means':rows,'exact_seed_rows':exact,'by_alpha_shots':[],'exact_by_alpha':[],'paired_tests':[]}
    for alpha in protocol['alphas']:
        group=[r for r in exact if r['alpha']==alpha]
        summary['exact_by_alpha'].append({'alpha':alpha,'statistics':{k:stats([r[k] for r in group]) for k in group[0] if k not in ['seed','alpha','angles']}})
        for n in protocol['shot_counts']:
            group=[r for r in rows if r['alpha']==alpha and r['shots']==n]
            summary['by_alpha_shots'].append({'alpha':alpha,'shots':n,'statistics':{k:stats([r['metrics'][k] for r in group]) for k in keys},
              'defined_repetitions':{k:sum(r['defined_repetitions'].get(k,100) for r in group) for k in keys}})
    for key in ['unseen_success','fm_minimum_hit','energy_rmse']:
        for n in [10,1000]:
            for alpha in protocol['alphas']:
                if alpha==5:continue
                diffs=[next(r['metrics'][key] for r in rows if r['seed']==seed and r['alpha']==alpha and r['shots']==n)-
                  next(r['metrics'][key] for r in rows if r['seed']==seed and r['alpha']==5 and r['shots']==n) for seed in protocol['seeds']]
                p=float(wilcoxon(diffs,method='auto').pvalue) if np.any(np.array(diffs)!=0) else 1.
                summary['paired_tests'].append({'metric':key,'alpha':alpha,'reference_alpha':5,'shots':n,'paired_differences':diffs,
                  'median_difference':float(np.median(diffs)),'p_raw':p,'p_bonferroni':min(1.,p*72),'family_size':72})
    save(OUT/'json/summary.json',summary)
    save(OUT/'json/artifact_verification.json',{'status':'passed','conditions':verified,'alpha_models':len(exact),
      'checks':'all raw shots decoded,independent candidate ranking/true lookup,input/model/checkpoint hashes,angle selection9point,lambda=Salpha,budget21; state norm/cost and alpha5 source parity checked in runner',
      'source_sha256':{str(seed):sha(OUT/f'json/seed_{seed}.json') for seed in protocol['seeds']}})
    report(summary);plot(summary,protocol);print(f'verified {verified} conditions at {len(exact)} model-alpha settings',flush=True)

def report(d):
    fmt=lambda s:'未定義' if s['median'] is None else f"{s['median']:.5g} [{s['q1']:.5g}, {s['q3']:.5g}]"
    lines=['# 標準QAOAのペナルティ感度','',
      '初期20件・同じ5固定FM、23量子ビットp1。13α（SAの12条件＋基準α5）、各αで同じ9点の厳密期待値角度選択。shots10,30,100,300,1000、各100反復、採用1件。全32500測定条件。新規OpenQARP計算と理想測定。',
      '', '中央値 [Q1,Q3] は5モデル間。測定指標はモデル内100反復の平均、エネルギー誤差はモデル内RMSE。候補0件の品質は未定義で除算しない。defined_seeds/repetitionsをJSONに記録。',
      '', '| α | 理想制約充足確率 | 新規1件獲得頻度（10shots） | 新規1件獲得頻度（1000shots） | Cost RMSE（10shots） | Cost RMSE（1000shots） |','|---:|---:|---:|---:|---:|---:|']
    for exact in d['exact_by_alpha']:
        a=exact['alpha'];r10=next(r for r in d['by_alpha_shots'] if r['alpha']==a and r['shots']==10)
        r1000=next(r for r in d['by_alpha_shots'] if r['alpha']==a and r['shots']==1000)
        vals=[exact['statistics']['exact_feasible_probability'],r10['statistics']['unseen_success'],r1000['statistics']['unseen_success'],r10['statistics']['energy_rmse'],r1000['statistics']['energy_rmse']]
        lines.append('| '+str(a)+' | '+' | '.join(fmt(s) for s in vals)+' |')
    lines+=['', 'Cost=(baseFM+λP)/S、λ=Sα。大きなαでCost自体のスケールが増えるため、RMSE増加だけで探索精度の悪化とは言えない。baseFM単独の平均推定RMSEと、Cost係数最大絶対値で割ったRMSEも別に保存・図示。baseFM単独は制約違反状態の外挿値も含み、候補の真値品質とは別。',
      '', 'α5は元の同一9点探索を再利用して選択状態を再計算し、前のショット研究とrawショット・エネルギー誤差が一致。その他は新規9点探索。9点の最小期待値が選ばれたことを確認したが、角度の大域最適性は証明していない。',
      '', '72ペア検定をBonferroni補正。5モデルの小標本。実機ノイズ・有限ショット角度最適化・BBO再学習・量子優位性は未評価。']
    (OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')

def plot(d,p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#777777','#9467bd','#1f77b4','#ff7f0e','#d62728'];markers=['v','s','o','^','D']
    panels=[('new_candidate_success','unseen_success','P(new feasible candidate)'),('fm_minimum_capture','fm_minimum_hit','P(best-unseen FM solution)'),
      ('selected_fm_gap','selected_fm_gap','Selected candidate FM gap (eV)'),('energy_rmse','energy_rmse','Normalized penalized Cost RMSE'),
      ('scaled_energy_rmse','coefficient_scaled_energy_rmse','Coefficient-scaled Cost RMSE'),('true_regret','normalized_regret_initial_plus_proposals','Regret: initial + selected one')]
    def setup(ax,label):
        ax.set_xscale('symlog',linthresh=.01);ax.set_xticks([0,.01,.1,1,10,100,1000],['0','.01','.1','1','10','100','1000'])
        ax.set_xlabel('Normalized penalty coefficient alpha');ax.set_ylabel(label);ax.grid(axis='y',alpha=.25)
    def panel(ax,key,label):
        for n,color,marker in zip(p['shot_counts'],colors,markers):
            group=[r for r in d['by_alpha_shots'] if r['shots']==n];x=[r['alpha'] for r in group];st=[r['statistics'][key] for r in group]
            arr=lambda k:np.array([np.nan if s[k] is None else s[k] for s in st])
            ax.plot(x,arr('median'),color=color,marker=marker,ms=4,label=f'{n} shots');ax.fill_between(x,arr('q1'),arr('q3'),color=color,alpha=.07)
        setup(ax,label);ax.legend(fontsize=8,ncol=2,loc='best')
        if label.startswith('P('):ax.set_ylim(-.025,1.025)
        if key in ['energy_rmse','coefficient_scaled_energy_rmse','base_FM_energy_rmse']:ax.set_yscale('log')
    def export(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{ext}/{name}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(16,8.5),layout='constrained')
    for ax,(_,key,label) in zip(axs.flat,panels):panel(ax,key,label)
    export(fig,'standard_qaoa_penalty_overview')
    for name,key,label in panels+[('selected_true_quality','selected_true_gap','Selected candidate true objective (eV)'),('base_FM_precision','base_FM_energy_rmse','Normalized base-FM energy RMSE')]:
        fig,ax=plt.subplots(figsize=(6.4,4.5),layout='constrained');panel(ax,key,label);export(fig,name)
    for name,key,label in [('ideal_feasibility','exact_feasible_probability','Ideal feasible probability'),('ideal_top5_mass','exact_unseen_FM_top5pct_probability','P(new FM top-5% sample)')]:
        fig,ax=plt.subplots(figsize=(6.4,4.5),layout='constrained');group=d['exact_by_alpha'];x=[r['alpha'] for r in group];st=[r['statistics'][key] for r in group]
        ax.plot(x,[s['median'] for s in st],color='#d62728',marker='^',label='Standard QAOA: median')
        ax.fill_between(x,[s['q1'] for s in st],[s['q3'] for s in st],color='#d62728',alpha=.15,label='IQR across5 models')
        setup(ax,label);ax.set_yscale('log');ax.legend(fontsize=8);export(fig,name)

if __name__=='__main__':main()
