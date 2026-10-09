"""Aggregate all predeclared paired seeds; retain failures and missingness."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

OUT=Path(__file__).resolve().parents[1]

def stats(values):
    present=[v for v in values if v is not None]
    if not present:return {'n':0,'median':None,'q25':None,'q75':None}
    a=np.array(present,dtype=float);q=np.quantile(a,[0.25,0.5,0.75])
    return {'n':len(a),'median':float(q[1]),'q25':float(q[0]),'q75':float(q[2])}


def main():
    path=OUT/'json/protocol.json';p=json.loads(path.read_text());assert p['status']=='ready'
    expected_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    rows=[]
    for seed in p['seeds']:
        source=OUT/f'json/seed_{seed}.json'
        rows.append(json.loads(source.read_text()) if source.exists() else {'seed':seed,'status':'missing','methods':{}})
    for r in rows:
        if r['status']=='completed':
            assert r['protocol_sha256']==expected_sha
            assert set(r['methods'])==set(p['methods'])
            for m in r['methods'].values():
                assert m['initial_sha256']==r['initial_sha256'] and m['fixed_model_sha256']==r['fixed_model_sha256']
                assert m['raw_sample_count']==1000 and m['bb_evaluations_initial']==20
                assert m['bb_evaluations_new']==m['accepted_count']<=5
                assert m['accepted_count']+m['unused_new_bb_budget']==5
                assert m['raw_invalid_count']+m['initial_duplicate_count']+m['unique_unseen_candidates']+m['within_unseen_pool_duplicate_count']==1000
                assert len(set(a['candidate_id'] for a in m['accepted_candidates']))==m['accepted_count']
                assert not set(a['candidate_id'] for a in m['accepted_candidates']) & set(r['initial_dataset']['candidate_ids'])
    metrics=['empirical_raw_feasible_rate','raw_unseen_feasible_fraction','accepted_count','unused_new_bb_budget','normalized_regret_initial_plus_proposals','best_initial_plus_proposals','proposal_best_hse_gap','empirical_conditional_unseen_mean_true_gap','generation_seconds','initial_duplicate_fraction','within_unseen_pool_duplicate_fraction','exact_feasible_probability','exact_unseen_feasible_probability','exact_conditional_unseen_expected_true_gap']
    summary={'scope':p['scope'],'protocol_sha256':expected_sha,'planned_seed_count':len(rows),'completed_seed_count':sum(r['status']=='completed' for r in rows),
      'failed_or_missing_seeds':[{'seed':r['seed'],'status':r['status'],'failure':r.get('failure')} for r in rows if r['status']!='completed'],
      'methods':{},'fm_quality':{},'paired_tests':[],'per_seed':[]}
    for method in p['methods']:
        summary['methods'][method]={key:stats([r['methods'].get(method,{}).get(key) for r in rows]) for key in metrics}
        summary['methods'][method]['failed_or_missing_runs']=sum(method not in r['methods'] for r in rows)
    for key in ['train_rmse','test_rmse','test_spearman']:
        summary['fm_quality'][key]=stats([r.get('fm_quality',{}).get(key) for r in rows])
    for metric in ['empirical_raw_feasible_rate','normalized_regret_initial_plus_proposals','generation_seconds']:
        for first,second in itertools.combinations(p['methods'],2):
            pairs=[(r['seed'],r['methods'][first][metric]-r['methods'][second][metric]) for r in rows if r['status']=='completed' and first in r['methods'] and second in r['methods']]
            diff=np.array([v for _,v in pairs]);rawp=float(wilcoxon(diff,alternative='two-sided',zero_method='wilcox',method='auto').pvalue) if np.any(diff) else (1.0 if len(diff) else None)
            summary['paired_tests'].append({'metric':metric,'first_minus_second':[first,second],'n_pairs':len(diff),'seed_differences':pairs,
              'difference_summary':stats(diff.tolist()),'wilcoxon_two_sided_p':rawp,'bonferroni_p_six_pairs':min(1,rawp*6) if rawp is not None else None,
              'power_note':'five-pair pilot; do not infer equivalence or general superiority from nonsignificance'})
    for r in rows:
        summary['per_seed'].append({'seed':r['seed'],'status':r['status'],'fm_quality':r.get('fm_quality'),'methods':{m:{k:r['methods'][m].get(k) for k in metrics} for m in r['methods']}})
    summary['initial_category_coverage']=[{'seed':r['seed'],'unique_organic_in_training':len(set(v['categories'][0] for v in r.get('initial_dataset',{}).get('records',[]))),'unique_cation_in_training':len(set(v['categories'][1] for v in r.get('initial_dataset',{}).get('records',[]))),'unique_anion_in_training':len(set(v['categories'][2] for v in r.get('initial_dataset',{}).get('records',[])))} for r in rows]
    (OUT/'json/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    def fmt(obj):
        if obj['median'] is None:return '未測定'
        return f"{obj['median']:.4g} [{obj['q25']:.4g}, {obj['q75']:.4g}]"
    report='''# Perovskites固定FM：候補生成の初期診断

これは固定した1つのFMに対する候補生成の比較。評価→再学習を繰り返すBBO比較ではない。事前条件は[protocol](../json/protocol.json)、全Seedの生記録はjson/seed_*.json、学習済みFMはdata/checkpoints/に保存した。

## 実施条件と集計

5 Seed、初期20候補を一様・非復元抽出。各Seedでrank=2、Adam、120 epochs、lr=0.1、CPU float32、raw eV、validationなしでFMを1回学習した。4手法は同じ初期データと同じFMを使う。ホールドアウトは残り172候補。全表は診断用の評価にのみ用い、角度選択や候補順位づけに真値を渡していない。

QAOAはp=1、OpenQARPの理想23量子ビット全状態ベクトル、3×3角度探索、1,000ショット。SAは4×250 reads、各1,000 sweeps。Adaptiveは各バッチ後にalphaを更新し、FMは固定。LargePenaltyはalpha=100、Penalty-QAOAはalpha=5、XYはRing・積W・lambda=0。SはベースFMのIsing係数から求め、内部lambda=S*alpha。QAOA CostはSで割って同じ角度探索を使う。

1,000サンプルから無効・初期既評価・プール内重複を除き、FM予測のよい未評価候補を最大5件採用。不足は補充せず記録する。候補確認の上限は各手法で初期20+最大5を共通にした。不足も除外しない。JSONのBB件数はこの診断で確認対象にした候補の論理的な件数であり、閉ループBBOで新たに実行した計算回数ではない。表引き済みの真値は診断用配列から参照する。サンプル数を揃えてもSAとQAOAの計算資源・時間が同等とは限らない。

数値は中央値 [Q1, Q3]。Raw feasibleは全方式で1,000サンプルからの実測比率を示す。QAOAの厳密な理想確率は各Seedの生JSONに別途保存した。

| 手法 | Raw feasible | 未評価・実行可能の比率 | 採用件数 | 初期＋採用候補の最良eV | 同最良値のNormalized Regret | 候補生成秒 |
|---|---:|---:|---:|---:|---:|---:|
'''
    for m in p['methods']:
        d=summary['methods'][m];report+='| '+m+' | '+' | '.join(fmt(d[k]) for k in ['empirical_raw_feasible_rate','raw_unseen_feasible_fraction','accepted_count','best_initial_plus_proposals','normalized_regret_initial_plus_proposals','generation_seconds'])+' |\n'
    report+=f"\n予定{len(rows)} Seed、完了{summary['completed_seed_count']} Seed。失敗・未完了: {summary['failed_or_missing_seeds']}。\n"
    report+='\n## FM自体の品質\n\n'
    for k,v in summary['fm_quality'].items():report+=f'- {k}: {fmt(v)}\n'
    report+='''
固定FMが真値を正しく順位づけできない場合、FMの最小値を見つけても実際の最良材料にはならない。全192候補の予測・ホールドアウトのRMSE/Spearmanと、候補探索の品質を分けて解釈する。

Normalized Regretは監査済みの固定表の最小値1.5249、最大値6.3242を使用。初期20件と実際に採用した最大5件の最良値について計算した。初期データだけでよい候補があるSeedでは、候補提案が改善しなくても低いRegretになるため、proposal_best_hse_gapも併記する。

## ペア差分と限界

同じSeedのペア差分、二側Wilcoxon、指標ごとの6ペアに対するBonferroni補正を[summary](../json/summary.json)に保存。5 Seedの予備検討では検定力が小さく、非有意を同等性の証明とはしない。FM誤差と真値選択の因果関係、量子優位性、閉ループBBOの優劣は結論しない。

QAOAは固定9点の探索と理想状態・有限ショット。実機のW-state準備ゲートやノイズは含まない。使用した学習条件・角度範囲を結果を見て変更していない。

## 確認できる保存物

- json/seed_*.json: 初期候補とハッシュ、FMハッシュ、QUBO、全サンプル、採用候補と真値、alpha更新、角度探索、時間。
- json/environment.json: Python・パッケージ・コードSHA256・プロトコルSHA256。
- json/summary.json: 全Seedと失敗の集計、中央値・IQR・ペア差分・検定。
- md/prevalidation.log: 小規模の標準X回路、FM→QUBO、分割エネルギー、重複・予算処理の検証。
'''
    (OUT/'md/RESULTS.md').write_text(report)
    print(json.dumps({'completed_seeds':summary['completed_seed_count'],'fm_quality':summary['fm_quality'],'methods':summary['methods']},ensure_ascii=False))


if __name__=='__main__':main()
