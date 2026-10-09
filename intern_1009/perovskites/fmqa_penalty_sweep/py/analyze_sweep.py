"""Verify saved raw samples, aggregate paired models, and produce export figures."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from run_sweep import OUT, PILOT, PerovskitesEvaluator, measure, sha, save

METRICS=['empirical_raw_feasible_rate','batch_any_feasible','batch_any_unseen_feasible',
 'batch_at_least_five_unseen','unique_unseen_candidates','accepted_count',
 'raw_unseen_feasible_fraction','initial_duplicate_fraction','within_unseen_pool_duplicate_fraction',
 'conditional_unseen_effective_diversity','conditional_unseen_mean_surrogate_gap','best_unseen_surrogate_gap',
 'raw_unseen_fm_top5pct_fraction','raw_unseen_true_top5pct_fraction','batch_any_unseen_fm_top5pct',
 'batch_any_unseen_true_top5pct','unique_unseen_true_top5pct_count',
 'empirical_conditional_unseen_mean_true_gap','proposal_best_hse_gap',
 'normalized_regret_initial_plus_proposals','generation_seconds']

def aggregate():
    protocol=json.loads((OUT/'json/protocol.json').read_text())
    bb=PerovskitesEvaluator();cats=[list(c) for c in bb.candidates()]
    patterns=np.array([bb.encode(c) for c in cats],dtype=np.int8)
    truth=np.array([bb.evaluate(c) for c in cats]);rows=[];verified=0
    for seed in protocol['seeds']:
        d=json.loads((OUT/f'json/seed_{seed}.json').read_text())
        base=json.loads((PILOT/f'json/seed_{seed}.json').read_text())
        assert d['status']=='completed' and d['protocol_sha256']==sha(OUT/'json/protocol.json')
        assert d['source_pilot_sha256']==sha(PILOT/f'json/seed_{seed}.json')
        assert d['checkpoint_sha256']==base['checkpoint_sha256']==sha(PILOT/base['checkpoint'])
        assert d['initial_sha256']==base['initial_sha256'] and d['fixed_model_sha256']==base['fixed_model_sha256']
        assert [c['alpha'] for c in d['conditions']]==protocol['alphas']
        q=np.array(base['qubo']);ids=base['initial_dataset']['candidate_ids']
        for c in d['conditions']:
            assert abs(c['lambda_internal']-c['alpha']*d['base_Ising_scale_S'])<1e-9
            assert len(c['runs'])==protocol['sampling_repetitions']
            for r in c['runs']:
                indices=np.array(r['raw_basis_indices'],dtype=np.uint64)
                assert len(indices)==1000 and np.all(indices<1<<23)
                bits=((indices[:,None]>>np.arange(23,dtype=np.uint64))&1).astype(np.int8)
                check=measure(bits,q,base['offset'],patterns,ids,truth,cats)
                for k in METRICS:
                    if k=='generation_seconds':continue
                    assert (r[k] is None and check[k] is None) or (r[k] is not None and check[k] is not None and abs(r[k]-check[k])<1e-10),(seed,c['alpha'],k)
                assert r['accepted_count']<=5
                verified+=1
            row={'seed':seed,'alpha':c['alpha'],'lambda_internal':c['lambda_internal'],'metrics':{},'defined_repetitions':{}}
            for k in METRICS:
                values=[r[k] for r in c['runs'] if r[k] is not None]
                row['metrics'][k]=float(np.mean(values)) if values else None
                row['defined_repetitions'][k]=len(values)
            rows.append(row)
    summary={'status':'completed','verified_repetitions':verified,'protocol_sha256':sha(OUT/'json/protocol.json'),
      'seed_level_repetition_means':rows,'by_alpha':[],'paired_tests':[]}
    for alpha in protocol['alphas']:
        selected=[r for r in rows if r['alpha']==alpha];stats={}
        for k in METRICS:
            vals=[r['metrics'][k] for r in selected if r['metrics'][k] is not None]
            stats[k]={'defined_model_seeds':len(vals),'defined_repetitions':sum(r['defined_repetitions'][k] for r in selected),
              'median':float(np.median(vals)) if vals else None,
              'q1':float(np.quantile(vals,.25)) if vals else None,'q3':float(np.quantile(vals,.75)) if vals else None}
        summary['by_alpha'].append({'alpha':alpha,'statistics':stats})
    total=(len(protocol['alphas'])-1)*len(protocol['paired_test_metrics'])
    for metric in protocol['paired_test_metrics']:
        for alpha in protocol['alphas']:
            if alpha==100:continue
            diffs=[next(r['metrics'][metric] for r in rows if r['seed']==seed and r['alpha']==alpha)-
              next(r['metrics'][metric] for r in rows if r['seed']==seed and r['alpha']==100) for seed in protocol['seeds']]
            p=float(wilcoxon(diffs,alternative='two-sided',method='auto').pvalue) if np.any(np.array(diffs)!=0) else 1.
            summary['paired_tests'].append({'metric':metric,'alpha':alpha,'reference_alpha':100,'paired_differences':diffs,
              'median_difference':float(np.median(diffs)),'p_raw':p,'p_bonferroni':min(1.,p*total),'family_size':total})
    save(OUT/'json/summary.json',summary)
    save(OUT/'json/artifact_verification.json',{'status':'passed','verified_repetitions':verified,
      'checks':'all raw samples independently decoded and metrics recomputed; five source/checkpoint/initial/model identities; lambda=S alpha; full alpha/seed/rep coverage; no failed or excluded runs',
      'seed_files_sha256':{str(seed):sha(OUT/f'json/seed_{seed}.json') for seed in protocol['seeds']}})
    return summary

def report(summary):
    def fmt(s):
        return '未定義' if s['median'] is None else f"{s['median']:.4g} [{s['q1']:.4g}, {s['q3']:.4g}]"
    columns=['empirical_raw_feasible_rate','batch_at_least_five_unseen','unique_unseen_candidates',
      'conditional_unseen_mean_surrogate_gap','batch_any_unseen_true_top5pct','normalized_regret_initial_plus_proposals']
    lines=['# 固定FM・SAペナルティ感度の結果','',
      '23 One-Hot変数。初期20件で学習済みの5モデルを固定。12係数×10サンプリング反復×5モデル＝600条件、各1000サンプル。閉ループBBO・再学習は行っていない。',
      '', '各セルは、モデル内10反復の平均を求めた後、5モデル間の中央値 [Q1,Q3]。獲得確率は10バッチ反復中の成功頻度であり、理論的な厳密確率ではない。', '',
      '| α | Raw制約充足率 | 新規5解以上の獲得率 | 新規ユニーク解数 | 条件付き平均FM gap (eV) | 新規真値Top5%獲得率 | 初期＋提案Regret |',
      '|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary['by_alpha']:
        lines.append('| '+str(row['alpha'])+' | '+' | '.join(fmt(row['statistics'][c]) for c in columns)+' |')
    lines += ['', 'FM gapはペナルティを含まない予測値と、実行可能192候補の予測最小値との差。新規実行可能解がない反復では未定義とし、0で埋めない。以下に定義可能件数を示す。真値Top5%は192候補中10件で、初期データの再生成は新規獲得に数えない。', '',
      '| α | FM gapが定義できた反復数 / 50 | モデル数 / 5 |', '|---:|---:|---:|']
    for row in summary['by_alpha']:
        s=row['statistics']['conditional_unseen_mean_surrogate_gap']
        lines.append(f"| {row['alpha']} | {s['defined_repetitions']} | {s['defined_model_seeds']} |")
    lines += ['', 'α=100を基準に3事前指定指標×11係数＝33比較のペア差分とWilcoxon検定をsummary.jsonに保存。全33比較にBonferroni補正。5モデルの小標本であり、非有意は同等性を証明しない。',
      '', '全600条件のrawビット列を再デコードし、保存指標と照合。モデル・初期データ・チェックポイントの同一性とλ=Sα、予算上限を確認。起動時に候補一覧APIの呼出し誤りがあり、サンプリング開始前に修正した。起動失敗ログも保持した。',
      '', '[図の説明](FIGURES.md) / [再実行手順](REPRODUCIBILITY.md) / [機械可読集計](../json/summary.json)']
    (OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')

PANELS=[
 ('raw_feasibility','empirical_raw_feasible_rate','Raw feasible fraction'),
 ('five_candidate_success','batch_at_least_five_unseen','P(at least 5 new feasible candidates)'),
 ('unique_candidates','unique_unseen_candidates','Unique new feasible candidates'),
 ('surrogate_quality','conditional_unseen_mean_surrogate_gap','Conditional mean FM gap (eV)'),
 ('true_top5_success','batch_any_unseen_true_top5pct','P(new true top-5% candidate)'),
 ('true_regret','normalized_regret_initial_plus_proposals','Regret: initial + proposed'),
 ('effective_diversity','conditional_unseen_effective_diversity','Conditional effective diversity'),
 ('duplicates','within_unseen_pool_duplicate_fraction','Raw new-pool duplicate fraction'),
 ('generation_time','generation_seconds','Candidate generation time (s)')]

def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    alphas=np.array([r['alpha'] for r in summary['by_alpha']])
    def panel(ax,key,label):
        for seed in [42,101,2024,7,19]:
            vals=[next(r['metrics'][key] for r in summary['seed_level_repetition_means'] if r['seed']==seed and r['alpha']==a) for a in alphas]
            ax.plot(alphas,[np.nan if v is None else v for v in vals],color='#1f77b4',alpha=.22,lw=.8)
        stats=[r['statistics'][key] for r in summary['by_alpha']]
        med=np.array([np.nan if s['median'] is None else s['median'] for s in stats])
        lo=np.array([np.nan if s['q1'] is None else s['q1'] for s in stats]);hi=np.array([np.nan if s['q3'] is None else s['q3'] for s in stats])
        ax.plot(alphas,med,'o-',color='#1f77b4',label='Fixed-alpha SA: median')
        ax.fill_between(alphas,lo,hi,color='#1f77b4',alpha=.16,label='IQR across 5 models')
        ax.set_xscale('symlog',linthresh=.01);ax.set_xticks([0,.01,.1,1,10,100,1000],['0','.01','.1','1','10','100','1000'])
        ax.set_xlabel('Normalized penalty coefficient alpha');ax.set_ylabel(label)
        ax.grid(axis='y',alpha=.25);ax.legend(fontsize=8,loc='best')
        if 'fraction' in label or label.startswith('P('):ax.set_ylim(-.025,1.025)
        if key=='generation_seconds':ax.set_yscale('log')
    def export(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{ext}/{name}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(16,8.5),layout='constrained')
    for ax,(_,key,label) in zip(axs.flat,PANELS[:6]):panel(ax,key,label)
    export(fig,'penalty_sensitivity_overview')
    for name,key,label in PANELS:
        fig,ax=plt.subplots(figsize=(6.4,4.5),layout='constrained');panel(ax,key,label);export(fig,name)

if __name__=='__main__':
    summary=aggregate();report(summary);plot(summary)
