"""Nested subsampling of saved SA samples, top1 selection; not fresh SA or BBO."""
import sys,json,platform
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

OUT=Path(__file__).resolve().parents[1];PROBLEM=OUT.parent
SOURCE=PROBLEM/'fmqa_penalty_sweep';PILOT=PROBLEM/'fixed_fm_pilot';TOP1=PROBLEM/'fmqa_penalty_sweep_top1'
sys.path[:0]=[str(SOURCE/'py'),str(PILOT/'py')]
from run_sweep import PerovskitesEvaluator, sha, save
from run_pilot import score_candidates

KEYS=['empirical_raw_feasible_rate','unseen_success','unique_unseen_candidates','selected_fm_gap',
      'fm_minimum_hit','selected_true_gap','normalized_regret_initial_plus_proposals','selected_true_top5_hit']

def main():
    protocol=json.loads((OUT/'json/protocol.json').read_text());assert protocol['status']=='ready'
    assert json.loads((SOURCE/'json/artifact_verification.json').read_text())['status']=='passed'
    assert json.loads((TOP1/'json/artifact_verification.json').read_text())['status']=='passed'
    bb=PerovskitesEvaluator();cats=[list(c) for c in bb.candidates()]
    patterns=np.array([bb.encode(c) for c in cats],dtype=np.int8);truth=np.array([bb.evaluate(c) for c in cats])
    lookup={tuple(p):i for i,p in enumerate(patterns)};true_top=set(np.argsort(truth,kind='stable')[:10].tolist())
    rows=[];verified=0;source_hashes={}
    for seed in protocol['seeds']:
        path=SOURCE/f'json/seed_{seed}.json';raw=json.loads(path.read_text());source_hashes[str(seed)]=sha(path)
        base=json.loads((PILOT/f'json/seed_{seed}.json').read_text());prior=json.loads((TOP1/f'json/seed_{seed}.json').read_text())
        assert raw['status']=='completed' and raw['source_pilot_sha256']==sha(PILOT/f'json/seed_{seed}.json')
        assert raw['checkpoint_sha256']==sha(PILOT/base['checkpoint'])
        assert raw['initial_sha256']==base['initial_sha256']==prior['initial_sha256']
        assert raw['fixed_model_sha256']==base['fixed_model_sha256']==prior['fixed_model_sha256']
        assert raw['protocol_sha256']==sha(SOURCE/'json/protocol.json')
        q=np.array(base['qubo']);offset=base['offset'];initial=base['initial_dataset']['candidate_ids']
        pred=np.einsum('bi,ij,bj->b',patterns,q,patterns)+offset
        unseen_ids=[i for i in range(192) if i not in set(initial)];unseen_min=pred[unseen_ids].min()
        orders=[np.random.default_rng(np.random.SeedSequence([seed,rep,1009])).permutation(1000) for rep in range(10)]
        assert all(len(set(order))==1000 for order in orders)
        record={'seed':seed,'status':'completed','source_sha256':sha(path),'top1_source_sha256':sha(TOP1/f'json/seed_{seed}.json'),
          'initial_sha256':raw['initial_sha256'],'fixed_model_sha256':raw['fixed_model_sha256'],
          'protocol_sha256':sha(OUT/'json/protocol.json'),'subset_position_orders':[o.tolist() for o in orders],'conditions':[]}
        assert [c['alpha'] for c in raw['conditions']]==protocol['alphas']
        for condition in raw['conditions']:
            alpha=condition['alpha'];assert len(condition['runs'])==10
            reference=next(c for c in prior['conditions'] if c['alpha']==alpha)
            by_n={n:[] for n in protocol['sample_counts']}
            for rep,old in enumerate(condition['runs']):
                assert old['repetition']==rep and len(old['raw_basis_indices'])==1000
                full_indices=np.array(old['raw_basis_indices'],dtype=np.uint64);last_positions=set();last_fm=None
                for n in protocol['sample_counts']:
                    positions=orders[rep][:n];assert last_positions<=set(positions);last_positions=set(positions)
                    indices=full_indices[positions]
                    bits=((indices[:,None]>>np.arange(23,dtype=np.uint64))&1).astype(np.int8)
                    result=score_candidates(bits,{},q,offset,cats,patterns,initial,truth,{'accepted_candidates':1})
                    result['allowed_total_bb_budget']=21
                    selected=result['accepted_candidates'][0] if result['accepted_count'] else None
                    available={lookup[tuple(b)] for b in bits if tuple(b) in lookup}-set(initial)
                    expected=min(available,key=lambda i:(pred[i],tuple(patterns[i]))) if available else None
                    assert (selected['candidate_id'] if selected else None)==expected
                    if selected:
                        assert selected['hse_gap']==truth[expected]
                        if last_fm is not None:assert selected['fm_prediction']<=last_fm+1e-10
                        last_fm=selected['fm_prediction']
                    result.update({'repetition':rep,'sample_count':n,'unseen_success':int(selected is not None),
                      'selected_fm_gap':selected['surrogate_gap_to_global_feasible_fm_min'] if selected else None,
                      'selected_true_gap':selected['hse_gap'] if selected else None,
                      'fm_minimum_hit':int(selected is not None and abs(selected['fm_prediction']-unseen_min)<1e-10),
                      'selected_true_top5_hit':int(selected is not None and expected in true_top)})
                    assert result['accepted_count']<=1 and result['raw_sample_count']==n
                    assert result['raw_invalid_count']+result['initial_duplicate_count']+result['within_unseen_pool_duplicate_count']+result['unique_unseen_candidates']==n
                    assert result['bb_evaluations_initial']+result['bb_evaluations_new']<=21
                    if n==1000:
                        ref=reference['runs'][rep]
                        assert result['accepted_candidates']==ref['accepted_candidates']
                        assert result['best_initial_plus_proposals']==ref['best_initial_plus_proposals']
                        assert result['empirical_raw_feasible_rate']==ref['empirical_raw_feasible_rate']
                    by_n[n].append(result);verified+=1
            for n,runs in by_n.items():
                record['conditions'].append({'alpha':alpha,'sample_count':n,'lambda_internal':condition['lambda_internal'],'runs':runs})
                row={'seed':seed,'alpha':alpha,'sample_count':n,'metrics':{},'defined_repetitions':{}}
                for key in KEYS:
                    values=[r[key] for r in runs if r[key] is not None]
                    row['metrics'][key]=float(np.mean(values)) if values else None;row['defined_repetitions'][key]=len(values)
                rows.append(row)
        save(OUT/f'json/seed_{seed}.json',record)
    summary={'status':'completed','scope':protocol['scope'],'verified_conditions':verified,'protocol_sha256':sha(OUT/'json/protocol.json'),
      'source_sha256':source_hashes,'seed_level_repetition_means':rows,'by_alpha_and_count':[],'paired_tests':[]}
    for n in protocol['sample_counts']:
        for alpha in protocol['alphas']:
            group=[r for r in rows if r['sample_count']==n and r['alpha']==alpha];stats={}
            for key in KEYS:
                values=[r['metrics'][key] for r in group if r['metrics'][key] is not None]
                stats[key]={'median':float(np.median(values)) if values else None,'q1':float(np.quantile(values,.25)) if values else None,
                  'q3':float(np.quantile(values,.75)) if values else None,'defined_model_seeds':len(values),
                  'defined_repetitions':sum(r['defined_repetitions'][key] for r in group)}
            summary['by_alpha_and_count'].append({'sample_count':n,'alpha':alpha,'statistics':stats})
    for key in ['unseen_success','fm_minimum_hit','normalized_regret_initial_plus_proposals']:
        for alpha in protocol['alphas']:
            for n in protocol['sample_counts'][:-1]:
                diffs=[next(r['metrics'][key] for r in rows if r['seed']==seed and r['sample_count']==n and r['alpha']==alpha)-
                  next(r['metrics'][key] for r in rows if r['seed']==seed and r['sample_count']==1000 and r['alpha']==alpha) for seed in protocol['seeds']]
                p=float(wilcoxon(diffs,method='auto').pvalue) if np.any(np.array(diffs)!=0) else 1.
                summary['paired_tests'].append({'metric':key,'alpha':alpha,'sample_count':n,'reference_count':1000,
                  'paired_differences':diffs,'median_difference':float(np.median(diffs)),'p_raw':p,'p_bonferroni':min(1.,p*144),'family_size':144})
    save(OUT/'json/summary.json',summary)
    save(OUT/'json/artifact_verification.json',{'status':'passed','conditions':verified,
      'checks':'nested subsets; independent direct ranking; n1000 parity with top1; selected FM value monotone with n; initial/model/checkpoint identity; cap21; complete coverage and all failures retained'})
    save(OUT/'json/environment.json',{'python':sys.version,'platform':platform.platform(),'script_sha256':sha(Path(__file__)),
      'selector_sha256':sha(PILOT/'py/run_pilot.py'),'source_environment_sha256':sha(SOURCE/'json/environment.json')})
    report(summary);plot(summary,protocol)
    print(f'completed {verified} nested subset conditions; no fresh SA or BBO',flush=True)

def report(summary):
    def fmt(s):return '未定義' if s['median'] is None else f"{s['median']:.4g} [{s['q1']:.4g}, {s['q3']:.4g}]"
    columns=['unseen_success','fm_minimum_hit','selected_fm_gap','selected_true_gap','normalized_regret_initial_plus_proposals']
    lines=['# サンプル数比較の結果','',
      '保存済みSAの各1000サンプルを無作為に並べ替え、先頭10・30・100・300・1000個から最小FM予測の未評価実行可能候補を1件採用。5モデル×12係数×10反復×5サンプル数＝3000条件の二次解析。SAを各num_reads設定で新規実行した結果ではない。',
      '', '値は各モデル内10反復平均の5モデル中央値 [Q1,Q3]。欠損の条件付き品質は0にせず、定義件数をsummary.jsonへ保存。獲得頻度は有限反復の実測頻度。',
      '', '| α | サンプル数 | 新規1解獲得頻度 | FM最小解獲得頻度 | 選択FM gap (eV) | 選択真値 (eV) | 初期＋1件Regret |','|---:|---:|---:|---:|---:|---:|---:|']
    for row in sorted(summary['by_alpha_and_count'],key=lambda r:(r['alpha'],r['sample_count'])):
        lines.append('| '+str(row['alpha'])+' | '+str(row['sample_count'])+' | '+' | '.join(fmt(row['statistics'][key]) for key in columns)+' |')
    lines += ['', 'FM最小解獲得は、172件の未評価実行可能空間の最小FM予測値に達した頻度（1e-10許容）。失敗は0を含む。選択FM gapは全192件の実行可能FM最小値との差。真値は事後診断だけに使用。',
      '', '全3000条件でraw再デコード、独立したFM順位照合、nested部分集合、1000件の既存結果との一致を検証。3指標×12係数×4サンプル数の144ペア検定をBonferroni補正。5モデルの小標本で、非有意は同等性ではない。',
      '', 'raw分布は元のSA計測。小さいnum_readsで新規SAを回した所要時間・独立な生成分布は未測定。閉ループBBOは未実施。']
    (OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')

def plot(summary,protocol):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#777777','#9467bd','#1f77b4','#ff7f0e','#2ca02c'];markers=['v','s','o','^','D']
    panels=[('raw_feasibility','empirical_raw_feasible_rate','Raw feasible fraction'),
      ('one_candidate_success','unseen_success','P(new feasible candidate)'),
      ('fm_minimum_capture','fm_minimum_hit','P(best-unseen FM solution)'),
      ('selected_fm_gap','selected_fm_gap','Selected candidate FM gap (eV)'),
      ('selected_true_quality','selected_true_gap','Selected candidate true objective (eV)'),
      ('true_regret','normalized_regret_initial_plus_proposals','Regret: initial + selected one')]
    def panel(ax,key,label):
        for n,color,marker in zip(protocol['sample_counts'],colors,markers):
            rows=sorted([r for r in summary['by_alpha_and_count'] if r['sample_count']==n],key=lambda r:r['alpha'])
            x=[r['alpha'] for r in rows];stats=[r['statistics'][key] for r in rows]
            values=lambda k:np.array([np.nan if s[k] is None else s[k] for s in stats])
            ax.plot(x,values('median'),marker=marker,color=color,lw=1.5,ms=4,label=f'{n} samples')
            ax.fill_between(x,values('q1'),values('q3'),color=color,alpha=.07)
        ax.set_xscale('symlog',linthresh=.01);ax.set_xticks([0,.01,.1,1,10,100,1000],['0','.01','.1','1','10','100','1000'])
        ax.set_xlabel('Normalized penalty coefficient alpha');ax.set_ylabel(label);ax.grid(axis='y',alpha=.25)
        ax.legend(fontsize=8,ncol=2,loc='best')
        if label.startswith('P(') or 'fraction' in label:ax.set_ylim(-.025,1.025)
    def export(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{ext}/{name}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(16,8.5),layout='constrained')
    for ax,(_,key,label) in zip(axs.flat,panels):panel(ax,key,label)
    export(fig,'sampling_budget_overview')
    for name,key,label in panels:
        fig,ax=plt.subplots(figsize=(6.4,4.5),layout='constrained');panel(ax,key,label);export(fig,name)

if __name__=='__main__':main()
