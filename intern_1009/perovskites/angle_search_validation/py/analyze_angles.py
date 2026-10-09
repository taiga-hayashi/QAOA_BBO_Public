"""Audit small-model angle search and plot matched-budget diagnostics."""
import json
import numpy as np
from scipy.stats import wilcoxon
from run_angles import OUT,PILOT,sha,save,all_bitstrings_lsb,one_hot_penalty,batch_best

METRICS=['feasible_probability','fm_optimum_probability','success10','fm_optimum_success10',
 'conditional_best10_FM_gap_eV','expectation_gap_to_penalized_ground','max_feasibility_in_evaluated_grid']

def stats(values):
    values=[v for v in values if v is not None]
    return {'median':float(np.median(values)) if values else None,'q1':float(np.quantile(values,.25)) if values else None,
      'q3':float(np.quantile(values,.75)) if values else None,'defined_seeds':len(values)}

def main():
    protocol=json.loads((OUT/'json/protocol.json').read_text());rows=[];calls=0;checked=0
    for seed in protocol['seeds']:
        d=json.loads((OUT/f'json/seed_{seed}.json').read_text());base=json.loads((PILOT/f'json/seed_{seed}.json').read_text())
        assert d['status']=='completed' and d['source_sha256']==sha(PILOT/f'json/seed_{seed}.json')
        assert d['model_sha256']==base['fixed_model_sha256'] and d['checkpoint_sha256']==sha(PILOT/base['checkpoint'])
        assert d['protocol_sha256']==sha(OUT/'json/protocol.json') and len(d['variants'])==2
        for v in d['variants']:
            assert len(v['settings'])==44
            bits=all_bitstrings_lsb(v['n']);q=np.array(v['restricted_qubo']);fm=np.einsum('bi,ij,bj->b',bits,q,bits)+v['offset']
            penalty=one_hot_penalty(bits,v['group_sizes']);valid=penalty==0;div=v['base_Ising_S'] if v['base_Ising_S']>0 else 1.
            for r in v['settings']:
                p=np.array(r['selected_basis_probabilities']);assert len(p)==1<<v['n'] and abs(p.sum()-1)<1e-10
                cost=(fm+r['lambda_internal']*penalty)/div;selected=min(r['angle_search'],key=lambda x:x['expected_common_baseS_Cost'])
                assert r['selected_angles']==[selected['gamma'],selected['beta']]
                assert abs(p@cost-r['expected_common_baseS_Cost'])<1e-8
                assert abs(p[valid].sum()-r['feasible_probability'])<1e-10
                gap,success=batch_best(p[valid],fm[valid],10)
                assert (gap is None and r['conditional_best10_FM_gap_eV'] is None) or abs(gap-r['conditional_best10_FM_gap_eV'])<1e-10
                assert abs(success-r['success10'])<1e-10
                if r['method']=='XY-FMQAOA':assert r['lambda_internal']==0 and abs(r['feasible_probability']-1)<1e-10
                else:
                    # Independent standard p1 cost-phase/RX probability check (global phase ignored).
                    gamma,beta=r['selected_angles'];ref=np.exp(-1j*gamma*cost/r['Cost_divisor'])/np.sqrt(len(bits))
                    for qubit in range(v['n']):
                        for i in range(len(ref)):
                            if i&(1<<qubit):continue
                            j=i|(1<<qubit);a,b=ref[i],ref[j]
                            ref[i]=np.cos(beta)*a-1j*np.sin(beta)*b;ref[j]=np.cos(beta)*b-1j*np.sin(beta)*a
                    np.testing.assert_allclose(np.abs(ref)**2,p,atol=1e-10,rtol=1e-8)
                assert len(r['angle_search'])==r['budget'];calls+=r['budget'];checked+=1
                rows.append({'seed':seed,'n':v['n'],'method':r['method'],'alpha':r['alpha'],'mode':r['mode'],'search':r['search'],
                  'budget':r['budget'],'metrics':{k:r[k] for k in METRICS},'selected_angles':r['selected_angles'],
                  'all_penalized_ground_states_feasible':r['all_penalized_ground_states_feasible']})
            # Refinement grids are nested; penalized expectation cannot increase except rounding.
            for key in {(r['method'],r['alpha'],r['mode']) for r in v['settings']}:
                chosen=[next(r for r in v['settings'] if (r['method'],r['alpha'],r['mode'])==key and r['search']==search) for search in ['wide9','wide49','wide361']]
                assert all(b['expected_common_baseS_Cost']<=a['expected_common_baseS_Cost']+1e-8 for a,b in zip(chosen,chosen[1:]))
    summary={'status':'completed','validated_settings':checked,'actual_circuit_evaluations':calls,'seed_rows':rows,'groups':[],'paired_tests':[]}
    keys=sorted({(r['n'],r['method'],r['alpha'],r['mode'],r['search']) for r in rows})
    for n,method,alpha,mode,search in keys:
        group=[r for r in rows if (r['n'],r['method'],r['alpha'],r['mode'],r['search'])==(n,method,alpha,mode,search)]
        assert len(group)==5
        summary['groups'].append({'n':n,'method':method,'alpha':alpha,'mode':mode,'search':search,'budget':group[0]['budget'],
          'statistics':{k:stats([r['metrics'][k] for r in group]) for k in METRICS},
          'ground_feasible_seed_count':sum(r['all_penalized_ground_states_feasible'] for r in group)})
    for n,method,alpha,mode in sorted({(r['n'],r['method'],r['alpha'],r['mode']) for r in rows}):
        for metric in ['feasible_probability','conditional_best10_FM_gap_eV']:
            diffs=[]
            for seed in protocol['seeds']:
                get=lambda search:next(r['metrics'][metric] for r in rows if (r['seed'],r['n'],r['method'],r['alpha'],r['mode'],r['search'])==(seed,n,method,alpha,mode,search))
                a,b=get('wide361'),get('legacy9')
                if a is not None and b is not None:diffs.append(a-b)
            p=float(wilcoxon(diffs,method='auto').pvalue) if diffs and np.any(np.array(diffs)!=0) else 1.
            summary['paired_tests'].append({'n':n,'method':method,'alpha':alpha,'mode':mode,'metric':metric,'contrast':'wide361_minus_legacy9',
              'paired_differences':diffs,'defined_pairs':len(diffs),'median_difference':float(np.median(diffs)) if diffs else None,
              'p_raw':p,'p_bonferroni':min(1.,p*44),'family_size':44})
    save(OUT/'json/summary.json',summary)
    save(OUT/'json/artifact_verification.json',{'status':'passed','settings':checked,'circuit_evaluations':calls,
      'checks':'restriction parity,all selected norms,independent X cost-phase/RX parity,XY no-penalty/feasible preservation,exact batch10 quality formula,grid minimum selection,nested refinement expectation monotonicity,complete5seed coverage'})
    report(summary);plot(summary);print(f'verified {checked} settings/{calls} circuit evaluations',flush=True)

def report(d):
    fmt=lambda s:'未定義' if s['median'] is None else f"{s['median']:.5g} [{s['q1']:.5g}, {s['q3']:.5g}]"
    lines=['# 角度探索の範囲・解像度・Costスケール検証','',
      '元の5固定FMを決めたカテゴリ規則で制限したN6（9実行可能候補）、N9（27候補）。再学習しない。標準α1,5,30,100,1000、baseSとfullCostSの2方式、XYはλ0。各探索の標準/XY回路評価数は9/9/49/361で共通。',
      '', f'全{d["validated_settings"]}設定、{d["actual_circuit_evaluations"]}回路評価。値は5モデル中央値 [Q1,Q3]。10ショットの獲得・選択品質は理想IIDの厳密式、実測反復ではない。初期データ重複除外はこの縮小回路診断では行わない。',
      '', '## 基準α5とXYの比較','',
      '| N | 方式 | 探索 | 回路評価数 | 制約充足確率 | 10shotsでFM最小解獲得 | 条件付き最良10shot FM gap (eV) |','|---:|---|---|---:|---:|---:|---:|']
    for n in [6,9]:
        for method,alpha,mode in [('Penalty-FMQAOA',5,'base_S'),('Penalty-FMQAOA',5,'full_Cost_S'),('XY-FMQAOA',0,'base_S')]:
            for search in ['legacy9','wide9','wide49','wide361']:
                r=next(r for r in d['groups'] if (r['n'],r['method'],r['alpha'],r['mode'],r['search'])==(n,method,alpha,mode,search))
                label='XY λ0' if method=='XY-FMQAOA' else 'X '+mode
                lines.append(f'| {n} | {label} | {search} | {r["budget"]} | '+' | '.join(fmt(r['statistics'][k]) for k in ['feasible_probability','fm_optimum_success10','conditional_best10_FM_gap_eV'])+' |')
    lines+=['','baseSでは元のFM係数尺度だけで回路を正規化。fullCostSはペナルティ込み最大Ising係数でも割る別対照。後者は物理gamma範囲/刻みを変更する再パラメータ化で、離散Costの順位やαを変えるものではない。',
      '', 'legacy9はγ=[0.05,0.4,0.8],β=[0.1,0.4,0.8]。wide9/49/361はγ∈[0,2π],β∈[0,π]の3/7/19点各軸。wide同士はnestedで、期待Cost最小値が非増加であることを確認。制約充足確率やFM品質の単調性は保証しない。',
      '', '全選択状態の標準確率を独立のphase/RX計算と照合。XYは全探索点で|1−Pfeasible|<1e-10、λ0。44ペア検定をBonferroni補正。5モデルの小標本。23変数の結果・実機・有限ショット角度探索・BBOへの一般化は未検証。']
    (OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')

def plot(d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    searches=['legacy9','wide9','wide49','wide361'];labels=['Legacy\n9 points','Wide\n9 points','Wide\n49 points','Wide\n361 points']
    styles=[('Penalty-FMQAOA',5,'base_S','#d62728','^','-','X: base-S'),
      ('Penalty-FMQAOA',5,'full_Cost_S','#8c1a1a','s','--','X: full-Cost-S'),('XY-FMQAOA',0,'base_S','#2ca02c','D','-','XY: lambda=0')]
    def panel(ax,n,key,label):
        for method,alpha,mode,color,marker,ls,title in styles:
            group=[next(r for r in d['groups'] if (r['n'],r['method'],r['alpha'],r['mode'],r['search'])==(n,method,alpha,mode,search)) for search in searches]
            st=[r['statistics'][key] for r in group];arr=lambda k:np.array([np.nan if s[k] is None else s[k] for s in st])
            ax.plot(range(4),arr('median'),color=color,marker=marker,ls=ls,label=title);ax.fill_between(range(4),arr('q1'),arr('q3'),color=color,alpha=.1)
        ax.set_xticks(range(4),labels);ax.set_ylabel(f'N={n}: {label}');ax.grid(axis='y',alpha=.25);ax.legend(fontsize=8,loc='best')
        if key in ['feasible_probability','fm_optimum_success10']:ax.set_ylim(-.025,1.025)
    def export(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{ext}/{name}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    specs=[('feasibility','feasible_probability','Feasible probability'),('optimum_capture','fm_optimum_success10','P(FM optimum in10 shots)'),
      ('candidate_quality','conditional_best10_FM_gap_eV','Conditional best10 FM gap (eV)')]
    fig,axs=plt.subplots(3,2,figsize=(13,12),layout='constrained')
    for row,(_,key,label) in zip(axs,specs):
        for ax,n in zip(row,[6,9]):panel(ax,n,key,label)
    export(fig,'angle_search_validation_overview')
    for name,key,label in specs:
        for n in [6,9]:
            fig,ax=plt.subplots(figsize=(6.4,4.5),layout='constrained');panel(ax,n,key,label);export(fig,f'{name}_N{n}')
    for n in [6,9]:
        fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
        for ax,mode in zip(axs,['base_S','full_Cost_S']):
            values=np.array([[next(r['statistics']['feasible_probability']['median'] for r in d['groups'] if
              (r['n'],r['method'],r['alpha'],r['mode'],r['search'])==(n,'Penalty-FMQAOA',a,mode,search)) for search in searches] for a in [1,5,30,100,1000]])
            im=ax.imshow(values,vmin=0,vmax=1,cmap='viridis',aspect='auto')
            ax.set_xticks(range(4),labels);ax.set_yticks(range(5),['1','5','30','100','1000']);ax.set_ylabel(f'N={n}, {mode}: alpha')
            for i in range(5):
                for j in range(4):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if values[i,j]<.5 else 'black',fontsize=9)
            fig.colorbar(im,ax=ax,label='Median feasible probability')
        export(fig,f'penalty_scale_feasibility_N{n}')

if __name__=='__main__':main()
