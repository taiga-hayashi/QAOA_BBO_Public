"""Aggregate ideal shot accuracy and output matched ten-candidate comparisons."""
import json,itertools
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from run_shots import OUT,PILOT,PerovskitesEvaluator,sha,save

KEYS=['empirical_raw_feasible_rate','unseen_success','fm_minimum_hit','selected_fm_gap','selected_true_gap',
      'normalized_regret_initial_plus_proposals','feasible_rate_abs_error']
STYLES={'Adaptive-FMQA':('#1f77b4','o'),'LargePenalty-FMQA':('#ff7f0e','s'),
        'Penalty-FMQAOA':('#d62728','^'),'XY-FMQAOA':('#2ca02c','D')}

def stats(vals):
    vals=[v for v in vals if v is not None]
    return {'median':float(np.median(vals)) if vals else None,'q1':float(np.quantile(vals,.25)) if vals else None,
      'q3':float(np.quantile(vals,.75)) if vals else None,'defined_seeds':len(vals)}

def main():
    protocol=json.loads((OUT/'json/protocol.json').read_text());rows=[];four=[];verified=0
    bb=PerovskitesEvaluator();cats=[list(c) for c in bb.candidates()];patterns=np.array([bb.encode(c) for c in cats],dtype=np.int8)
    lookup={tuple(p):i for i,p in enumerate(patterns)}
    for seed in protocol['seeds']:
        d=json.loads((OUT/f'json/seed_{seed}.json').read_text());base=json.loads((PILOT/f'json/seed_{seed}.json').read_text())
        assert d['status']=='completed' and d['protocol_sha256']==sha(OUT/'json/protocol.json')
        assert d['source_sha256']==sha(PILOT/f'json/seed_{seed}.json')
        assert d['fixed_model_sha256']==base['fixed_model_sha256'] and d['initial_sha256']==base['initial_sha256']
        pred=np.array(base['fm_quality']['all_candidate_predictions']);initial=set(base['initial_dataset']['candidate_ids'])
        def verify(r,n):
            indices=np.array(r['raw_basis_indices'],dtype=np.uint64);assert len(indices)==n
            bits=((indices[:,None]>>np.arange(23,dtype=np.uint64))&1).astype(np.int8)
            ids=[lookup.get(tuple(b)) for b in bits];available={i for i in ids if i is not None}-initial
            expected=min(available,key=lambda i:(pred[i],tuple(patterns[i]))) if available else None
            assert r['accepted_count']==int(expected is not None)
            if expected is not None:
                assert r['accepted_candidates'][0]['candidate_id']==expected
                assert r['selected_true_gap']==bb.evaluate(cats[expected])
            assert abs(r['empirical_raw_feasible_rate']-sum(i is not None for i in ids)/n)<1e-12
        for method,m in d['qaoa'].items():
            assert list(m['conditions'])==[str(n) for n in protocol['shot_counts']]
            for ntext,runs in m['conditions'].items():
                n=int(ntext);assert len(runs)==100
                for r in runs:verify(r,n);verified+=1
                row={'seed':seed,'method':method,'shots':n,'metrics':{},'defined_repetitions':{}}
                for k in KEYS:
                    vals=[r[k] for r in runs if r[k] is not None]
                    row['metrics'][k]=float(np.mean(vals)) if vals else None;row['defined_repetitions'][k]=len(vals)
                errors=np.array([r['energy_error'] for r in runs]);row['metrics'].update({
                  'energy_rmse':float(np.sqrt(np.mean(errors**2))),'energy_bias':float(errors.mean()),
                  'theoretical_energy_se':float(np.sqrt(m['exact_energy_variance']/n)),
                  'exact_unseen_batch_success':float(-np.expm1(n*np.log1p(-min(m['exact_unseen_probability'],1-1e-15)))),
                  'exact_fm_minimum_batch_success':float(-np.expm1(n*np.log1p(-min(m['exact_fm_minimum_probability'],1-1e-15))))})
                rows.append(row)
                if n==10:four.append(row)
        for method,runs in d['sa10'].items():
            assert len(runs)==100
            for r in runs:verify(r,10);verified+=1
            metrics={};defined={}
            for k in KEYS[:-1]:
                vals=[r[k] for r in runs if r[k] is not None];metrics[k]=float(np.mean(vals)) if vals else None;defined[k]=len(vals)
            four.append({'seed':seed,'method':method,'shots':10,'metrics':metrics,'defined_repetitions':defined})
    summary={'status':'completed','verified_repetitions':verified,'seed_rows':rows,'by_method_shots':[],
      'four_methods10_seed_rows':four,'four_methods10':[],'shot_paired_tests':[],'four_method_paired_tests':[]}
    allkeys=KEYS+['energy_rmse','energy_bias','theoretical_energy_se','exact_unseen_batch_success','exact_fm_minimum_batch_success']
    for method in protocol['methods']:
        for n in protocol['shot_counts']:
            group=[r for r in rows if r['method']==method and r['shots']==n]
            summary['by_method_shots'].append({'method':method,'shots':n,'statistics':{k:stats([r['metrics'][k] for r in group]) for k in allkeys},
              'defined_repetitions':{k:sum(r['defined_repetitions'].get(k,100) for r in group) for k in allkeys}})
    for method in STYLES:
        group=[r for r in four if r['method']==method]
        summary['four_methods10'].append({'method':method,'statistics':{k:stats([r['metrics'][k] for r in group]) for k in KEYS[:-1]},
          'defined_repetitions':{k:sum(r['defined_repetitions'][k] for r in group) for k in KEYS[:-1]}})
    def paired(values,family,extra):
        p=float(wilcoxon(values,method='auto').pvalue) if np.any(np.array(values)!=0) else 1.
        return {**extra,'paired_differences':values,'median_difference':float(np.median(values)),
          'p_raw':p,'p_bonferroni':min(1.,p*family),'family_size':family}
    for method in protocol['methods']:
        for k in ['unseen_success','fm_minimum_hit','energy_rmse']:
            for n in protocol['shot_counts'][:-1]:
                diffs=[next(r['metrics'][k] for r in rows if r['seed']==seed and r['method']==method and r['shots']==n)-
                  next(r['metrics'][k] for r in rows if r['seed']==seed and r['method']==method and r['shots']==1000) for seed in protocol['seeds']]
                summary['shot_paired_tests'].append(paired(diffs,24,{'method':method,'metric':k,'shots':n,'reference_shots':1000}))
    for a,b in itertools.combinations(STYLES,2):
        for k in ['empirical_raw_feasible_rate','unseen_success','normalized_regret_initial_plus_proposals']:
            diffs=[next(r['metrics'][k] for r in four if r['seed']==seed and r['method']==a)-
              next(r['metrics'][k] for r in four if r['seed']==seed and r['method']==b) for seed in protocol['seeds']]
            summary['four_method_paired_tests'].append(paired(diffs,18,{'method_a':a,'method_b':b,'metric':k}))
    save(OUT/'json/summary.json',summary)
    save(OUT/'json/artifact_verification.json',{'status':'passed','conditions':verified,
      'checks':'raw samples decoded; direct FM ranking and truth parity; full seed/method/shot/rep coverage; state norm,original exact probability/expectation parity and independent measured-energy checks in runner',
      'source_sha256':{str(seed):sha(OUT/f'json/seed_{seed}.json') for seed in protocol['seeds']}})
    report(summary);plot(summary)
    print(f'verified {verified} records and exported figures',flush=True)

def report(d):
    fmt=lambda s:'未定義' if s['median'] is None else f"{s['median']:.5g} [{s['q1']:.5g}, {s['q3']:.5g}]"
    cols=['unseen_success','fm_minimum_hit','selected_fm_gap','selected_true_gap','energy_rmse']
    lines=['# QAOA有限ショットの精度検証','','23量子ビットp1、既存の9点・厳密期待値探索で選んだ角度を固定して回路を再計算。5固定FM、各100測定反復、5ショット数、2方式。理想状態から新規に独立測定を模擬。ショットを使った角度再最適化ではない。',
      '', '各値はモデル内100反復平均（エネルギー誤差はRMSE）の、5モデル間中央値 [Q1,Q3]。条件付き品質は候補がある反復に限り、定義件数をJSONに保存。',
      '', '| 手法 | shots | 新規1解獲得頻度 | FM最小解獲得頻度 | 選択FM gap (eV) | 選択真値 (eV) | 正規化エネルギーRMSE |','|---|---:|---:|---:|---:|---:|---:|']
    for row in d['by_method_shots']:lines.append('| '+row['method']+' | '+str(row['shots'])+' | '+' | '.join(fmt(row['statistics'][k]) for k in cols)+' |')
    lines+=['','## 4手法・10サンプル／ショット・1件採用','','SAを10readsで新規計算。1000sweeps、4バッチ3,3,2,2。Adaptiveはバッチ間更新で、BBOの反復更新ではない。QAOAは上記固定角度の10shots。候補不足も保持。',
      '', '| 手法 | Raw制約充足率 | 新規1解獲得頻度 | 選択FM gap (eV) | 選択真値 (eV) | 初期＋1件Regret |','|---|---:|---:|---:|---:|---:|']
    for row in d['four_methods10']:lines.append('| '+row['method']+' | '+' | '.join(fmt(row['statistics'][k]) for k in ['empirical_raw_feasible_rate','unseen_success','selected_fm_gap','selected_true_gap','normalized_regret_initial_plus_proposals'])+' |')
    lines+=['','エネルギーは(baseFM+λP)/S。方法ごとにCostが異なるので、RMSEの大小をそのまま共通目的の優劣と解釈しない。理想IID測定の理論標準誤差sqrt(Var(E)/shots)もJSONと図に保存。',
      '', '同じ初期20件・固定FM・採用最大1件・上限21で統一。10出力は計算資源を揃えることではない。QAOAの角度探索は元の厳密期待値9点で、今回のショット予算に含めていない。ノイズ・実機・有限ショット角度探索・BBOは未検証。',
      '', '全6000条件のraw照合・集計。ショット数比較24検定、4手法10出力比較18検定で、それぞれBonferroni補正。5モデルの小標本、非有意は同等性を意味しない。']
    (OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')

def plot(d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    panels=[('raw_feasibility','empirical_raw_feasible_rate','Raw feasible fraction'),('one_candidate_success','unseen_success','P(new feasible candidate)'),
      ('fm_minimum_capture','fm_minimum_hit','P(best-unseen FM solution)'),('energy_precision','energy_rmse','Normalized energy RMSE'),
      ('selected_fm_gap','selected_fm_gap','Selected candidate FM gap (eV)'),('true_regret','normalized_regret_initial_plus_proposals','Regret: initial + selected one')]
    def panel(ax,key,label):
        for method in ['Penalty-FMQAOA','XY-FMQAOA']:
            group=[r for r in d['by_method_shots'] if r['method']==method];x=np.array([r['shots'] for r in group]);st=[r['statistics'][key] for r in group]
            arr=lambda k:np.array([np.nan if s[k] is None else s[k] for s in st]);color,marker=STYLES[method]
            ax.plot(x,arr('median'),color=color,marker=marker,label=method)
            ax.fill_between(x,arr('q1'),arr('q3'),color=color,alpha=.15)
            if key in ['unseen_success','fm_minimum_hit','energy_rmse']:
                ref={'unseen_success':'exact_unseen_batch_success','fm_minimum_hit':'exact_fm_minimum_batch_success','energy_rmse':'theoretical_energy_se'}[key]
                ax.plot(x,[r['statistics'][ref]['median'] for r in group],color=color,ls='--',lw=1,label=f'{"X" if method.startswith("Penalty") else "XY"} ideal reference')
        ax.set_xscale('log');ax.set_xticks([10,30,100,300,1000],['10','30','100','300','1000'])
        ax.set_xlabel('Measurement shots');ax.set_ylabel(label);ax.grid(axis='y',alpha=.25);ax.legend(fontsize=8,loc='best')
        if key in ['empirical_raw_feasible_rate','unseen_success','fm_minimum_hit']:ax.set_ylim(-.025,1.025)
        if key=='energy_rmse':ax.set_yscale('log')
    def export(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{ext}/{name}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(16,8.5),layout='constrained')
    for ax,(_,key,label) in zip(axs.flat,panels):panel(ax,key,label)
    export(fig,'qaoa_shot_accuracy_overview')
    for name,key,label in panels+[('selected_true_quality','selected_true_gap','Selected candidate true objective (eV)')]:
        fig,ax=plt.subplots(figsize=(6.4,4.5),layout='constrained');panel(ax,key,label);export(fig,name)
    def four_panel(ax,key,label):
        for i,row in enumerate(d['four_methods10']):
            s=row['statistics'][key];color,marker=STYLES[row['method']]
            if s['median'] is not None:ax.errorbar(i,s['median'],yerr=[[s['median']-s['q1']],[s['q3']-s['median']]],color=color,marker=marker,capsize=4,label=row['method'])
        ax.set_xticks(range(4),['Adaptive\nSA','Large-penalty\nSA','Penalty\nQAOA','XY\nQAOA']);ax.set_ylabel(label);ax.grid(axis='y',alpha=.25);ax.legend(fontsize=7)
        if key in ['empirical_raw_feasible_rate','unseen_success']:ax.set_ylim(-.025,1.025)
    fourpanels=[('empirical_raw_feasible_rate','Raw feasible fraction'),('unseen_success','P(new feasible candidate)'),('normalized_regret_initial_plus_proposals','Regret: initial + selected one')]
    fig,axs=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    for ax,(key,label) in zip(axs,fourpanels):four_panel(ax,key,label)
    export(fig,'four_methods_10_overview')
    for key,label in fourpanels:
        fig,ax=plt.subplots(figsize=(6.4,4.5),layout='constrained');four_panel(ax,key,label);export(fig,'four_methods_10_'+key)

if __name__=='__main__':main()
