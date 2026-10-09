"""User-requested top1 secondary analysis; reuse frozen SA pools and shared selector/plots."""
import sys,json,hashlib,platform
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

OUT=Path(__file__).resolve().parents[1];PROBLEM=OUT.parent
SOURCE=PROBLEM/'fmqa_penalty_sweep';PILOT=PROBLEM/'fixed_fm_pilot'
sys.path[:0]=[str(SOURCE/'py'),str(PILOT/'py')]
import analyze_sweep as previous
from run_sweep import PerovskitesEvaluator, sha, save
from run_pilot import score_candidates

def main():
    protocol=json.loads((OUT/'json/protocol.json').read_text());assert protocol['status']=='ready'
    assert json.loads((SOURCE/'json/artifact_verification.json').read_text())['status']=='passed'
    bb=PerovskitesEvaluator();cats=[list(c) for c in bb.candidates()]
    patterns=np.array([bb.encode(c) for c in cats],dtype=np.int8)
    truth=np.array([bb.evaluate(c) for c in cats]);lookup={tuple(p):i for i,p in enumerate(patterns)}
    true_top=set(np.argsort(truth,kind='stable')[:10].tolist())
    keys=previous.METRICS+['batch_at_least_one_unseen','selected_surrogate_gap','selected_true_gap','selected_true_top5_hit']
    rows=[];verified=0;source_hashes={};changed=0
    for seed in protocol['seeds']:
        path=SOURCE/f'json/seed_{seed}.json';source_hashes[str(seed)]=sha(path)
        raw=json.loads(path.read_text());base=json.loads((PILOT/f'json/seed_{seed}.json').read_text())
        assert raw['status']=='completed' and raw['initial_sha256']==base['initial_sha256']
        assert raw['fixed_model_sha256']==base['fixed_model_sha256']
        assert raw['checkpoint_sha256']==sha(PILOT/base['checkpoint'])
        assert raw['source_pilot_sha256']==sha(PILOT/f'json/seed_{seed}.json')
        assert raw['protocol_sha256']==sha(SOURCE/'json/protocol.json')
        assert [c['alpha'] for c in raw['conditions']]==protocol['alphas']
        q=np.array(base['qubo']);offset=base['offset'];initial=base['initial_dataset']['candidate_ids']
        predictions=np.einsum('bi,ij,bj->b',patterns,q,patterns)+offset
        record={'seed':seed,'status':'completed','source_sha256':sha(path),'protocol_sha256':sha(OUT/'json/protocol.json'),
          'initial_sha256':raw['initial_sha256'],'fixed_model_sha256':raw['fixed_model_sha256'],'conditions':[]}
        for condition in raw['conditions']:
            runs=[];assert len(condition['runs'])==10
            for old in condition['runs']:
                indices=np.array(old['raw_basis_indices'],dtype=np.uint64);assert len(indices)==1000
                bits=((indices[:,None]>>np.arange(23,dtype=np.uint64))&1).astype(np.int8)
                result=score_candidates(bits,{},q,offset,cats,patterns,initial,truth,{'accepted_candidates':1})
                result['allowed_total_bb_budget']=21
                # Raw distribution diagnostics are unchanged, reused with explicit source provenance.
                for k in keys:
                    if k not in result and k in old:result[k]=old[k]
                result.update({'repetition':old['repetition'],'generation_seconds':old['generation_seconds'],
                  'generation_seconds_provenance':'original SA measurement, no new sampling',
                  'batch_at_least_one_unseen':int(result['accepted_count']==1),
                  'selected_surrogate_gap':result['accepted_candidates'][0]['surrogate_gap_to_global_feasible_fm_min'] if result['accepted_count'] else None,
                  'selected_true_gap':result['proposal_best_hse_gap'],
                  'selected_true_top5_hit':int(result['accepted_count']==1 and result['accepted_candidates'][0]['candidate_id'] in true_top)})
                # Independent direct ranking check, then parity with the first originally selected candidate.
                available={lookup[tuple(b)] for b in bits if tuple(b) in lookup}-set(initial)
                expected=min(available,key=lambda i:(predictions[i],tuple(patterns[i]))) if available else None
                selected=result['accepted_candidates'][0]['candidate_id'] if result['accepted_count'] else None
                assert selected==expected
                assert selected==(old['accepted_candidates'][0]['candidate_id'] if old['accepted_count'] else None)
                assert result['raw_invalid_count']==old['raw_invalid_count'] and result['accepted_count']<=1
                assert result['bb_evaluations_initial']+result['bb_evaluations_new']<=21
                if selected is not None:assert result['selected_true_gap']==truth[selected]
                assert result['best_initial_plus_proposals']==min(truth[initial].min(),truth[selected] if selected is not None else float('inf'))
                changed+=int(abs(result['best_initial_plus_proposals']-old['best_initial_plus_proposals'])>1e-10)
                runs.append(result);verified+=1
            record['conditions'].append({'alpha':condition['alpha'],'lambda_internal':condition['lambda_internal'],'runs':runs})
            row={'seed':seed,'alpha':condition['alpha'],'metrics':{},'defined_repetitions':{}}
            for key in keys:
                vals=[r[key] for r in runs if r[key] is not None]
                row['metrics'][key]=float(np.mean(vals)) if vals else None;row['defined_repetitions'][key]=len(vals)
            rows.append(row)
        save(OUT/f'json/seed_{seed}.json',record)
    summary={'status':'completed','scope':protocol['scope'],'verified_repetitions':verified,'source_sha256':source_hashes,
      'protocol_sha256':sha(OUT/'json/protocol.json'),'seed_level_repetition_means':rows,'by_alpha':[],'paired_tests':[],
      'top1_vs_top5_changed_final_best_repetitions':changed}
    for alpha in protocol['alphas']:
        stats={};group=[r for r in rows if r['alpha']==alpha]
        for key in keys:
            vals=[r['metrics'][key] for r in group if r['metrics'][key] is not None]
            stats[key]={'median':float(np.median(vals)) if vals else None,'q1':float(np.quantile(vals,.25)) if vals else None,
              'q3':float(np.quantile(vals,.75)) if vals else None,'defined_model_seeds':len(vals),
              'defined_repetitions':sum(r['defined_repetitions'][key] for r in group)}
        summary['by_alpha'].append({'alpha':alpha,'statistics':stats})
    for key in ['empirical_raw_feasible_rate','batch_at_least_one_unseen','normalized_regret_initial_plus_proposals']:
        for alpha in protocol['alphas']:
            if alpha==100:continue
            diffs=[next(r['metrics'][key] for r in rows if r['seed']==seed and r['alpha']==alpha)-
              next(r['metrics'][key] for r in rows if r['seed']==seed and r['alpha']==100) for seed in protocol['seeds']]
            p=float(wilcoxon(diffs,method='auto').pvalue) if np.any(np.array(diffs)!=0) else 1.
            summary['paired_tests'].append({'metric':key,'alpha':alpha,'reference_alpha':100,'paired_differences':diffs,
              'median_difference':float(np.median(diffs)),'p_raw':p,'p_bonferroni':min(1.,p*33),'family_size':33})
    save(OUT/'json/summary.json',summary)
    save(OUT/'json/artifact_verification.json',{'status':'passed','verified_repetitions':verified,
      'checks':'independent direct ranking versus shared selector; old top5 first-candidate parity; raw feasibility unchanged; exact truth lookup; initial/model/checkpoint/source identity; budget cap21',
      'reanalysis_script_sha256':sha(Path(__file__))})
    save(OUT/'json/environment.json',{'python':sys.version,'platform':platform.platform(),
      'reused_environment_sha256':sha(SOURCE/'json/environment.json'),'source_protocol_sha256':sha(SOURCE/'json/protocol.json'),
      'code_sha256':{str(p):sha(p) for p in [Path(__file__),PILOT/'py/run_pilot.py',SOURCE/'py/analyze_sweep.py']}})
    columns=['empirical_raw_feasible_rate','batch_at_least_one_unseen','selected_surrogate_gap','selected_true_gap','normalized_regret_initial_plus_proposals']
    def fmt(s):return '未定義' if s['median'] is None else f"{s['median']:.5g} [{s['q1']:.5g}, {s['q3']:.5g}]"
    lines=['# 1件採用の再集計結果','','ユーザー指定により、元の1000サンプルからFM予測値が最も低い未評価実行可能候補を最大1件採用。全600条件を再集計。追加のSA計算・FM学習はしていない。5件採用の原実験は保存。',
      '', 'モデル内10反復平均の、5モデル間中央値 [Q1,Q3]。提案品質は候補が存在する反復だけで定義し、欠損を0にしない。',
      '', '| α | Raw制約充足率 | 1件獲得頻度 | 選択FM gap (eV) | 選択候補の真値 (eV) | 初期＋1件Regret |','|---:|---:|---:|---:|---:|---:|']
    for row in summary['by_alpha']:lines.append('| '+str(row['alpha'])+' | '+' | '.join(fmt(row['statistics'][k]) for k in columns)+' |')
    lines += ['',f'5件採用から1件採用に変えて初期＋提案の最良真値が変わった条件は600中{changed}件。詳しい候補とrawビット列は各seed JSON、定義可能件数と33補正ペア検定はsummary.json。',
      '', '初期20件＋新規最大1件の予算上限21。独立した再集計反復であり、連続BBOの評価回数ではない。採用順位には真値を使用しない。']
    (OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')
    previous.OUT=OUT
    previous.PANELS=[
      ('raw_feasibility','empirical_raw_feasible_rate','Raw feasible fraction'),
      ('one_candidate_success','batch_at_least_one_unseen','P(at least 1 new feasible candidate)'),
      ('unique_candidates','unique_unseen_candidates','Unique new feasible candidates'),
      ('selected_surrogate_quality','selected_surrogate_gap','Selected candidate FM gap (eV)'),
      ('selected_true_quality','selected_true_gap','Selected candidate true objective (eV)'),
      ('true_regret','normalized_regret_initial_plus_proposals','Regret: initial + selected one'),
      ('selected_true_top5_success','selected_true_top5_hit','P(selected true top-5% candidate)')]
    previous.plot(summary)
    print(f'completed {verified} top1 reanalyses; changed final best vs top5: {changed}',flush=True)

if __name__=='__main__':main()
