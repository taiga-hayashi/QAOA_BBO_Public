"""mlfilter-scale Phase 1 (exploratory): FMQA penalty sensitivity vs layer count.

  init     --layers L --truth-from <phase0 run>      new run dir (config: phase1_penalty.json + L)
  run      --run-dir D --seed S --condition ID --budget primary|fixed100
  analyze  --run-dir D                                summary over all finished runs in D
"""
import argparse
import hashlib
import json
import secrets
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

STUDY_DIR = Path(__file__).resolve().parents[1]
PROJECT = STUDY_DIR.parents[1]
CONFIG = json.loads((STUDY_DIR/'phase1_penalty.json').read_text())
PATHS = json.loads((PROJECT/'config/paths.local.json').read_text())['profiles']['server']
INTERN = Path(PATHS['intern_fujitsu_root'])
sys.path[:0] = [str(PROJECT/'src'), str(INTERN/'src'), str(INTERN), PATHS['extra_pythonpath']]
import mlfilter_fmqa as M

CODE = [Path(__file__).resolve(), STUDY_DIR/'phase1_penalty.json', PROJECT/'src/mlfilter_fmqa.py',
        *[INTERN/p for p in ['src/fm.py','src/fm_to_qubo.py','src/fmqa_solver.py','src/bb_function.py','src/qarp_backend.py',
                             'intern_0924/src/penalty.py']]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    tmp = Path(str(path)+'.tmp'); tmp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n'); tmp.replace(path)


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def code_hashes():
    return {str(p):sha(p) for p in CODE}


def init(layers, truth_from):
    truth_from = truth_from.resolve(); enum = json.loads((truth_from/'raw/enumeration.json').read_text())
    assert enum['status']=='completed' and enum['validation']=='passed' and enum['layers']==layers
    assert sha(truth_from/'raw/truth_table.npz')==enum['truth_table_sha256']
    canon = json.dumps({'config':CONFIG,'layers':layers},sort_keys=True,ensure_ascii=False,separators=(',',':'))
    config_hash = hashlib.sha256(canon.encode()).hexdigest(); base = PROJECT/'data/runs/mlfilter-scale'/('c-'+config_hash[:12])
    while True:
        run_id = 'r-'+secrets.token_hex(6)
        if not (base/run_id).exists(): break
    run_dir = base/run_id; (run_dir/'raw/bbo').mkdir(parents=True); (run_dir/'logs').mkdir()
    import platform, socket, importlib.metadata as md
    save(run_dir/'metadata.json', {'study':'mlfilter-scale','phase':'phase1-penalty','protocol_phase':'exploratory','layers':layers,
         'config_hash':config_hash,'config_id':base.name,'config_hash_method':'sha256 of canonical JSON {config: phase1_penalty.json, layers: L}',
         'run_id':run_id,'created_utc':now(),'code_sha256':code_hashes(),
         'truth_from':{'run_dir':str(truth_from.relative_to(PROJECT)),'truth_table_sha256':enum['truth_table_sha256'],
                       'optimum':enum['min_fom'],'worst':enum['max_fom'],'best_index':enum['best_index']},
         'environment':{'python':sys.version,'platform':platform.platform(),'hostname':socket.gethostname(),
                        'packages':{n:md.version(n) for n in ['numpy','torch','dimod','dwave-neal']}}})
    print(run_dir)


def load(run_dir):
    meta = json.loads((run_dir/'metadata.json').read_text())
    assert meta['code_sha256']==code_hashes(), 'code changed since init; create a new run'
    table = PROJECT/meta['truth_from']['run_dir']/'raw/truth_table.npz'
    assert sha(table)==meta['truth_from']['truth_table_sha256']
    return meta, np.load(table)['fom']


def run(run_dir, seed, cid, budget_kind):
    import torch
    torch.set_num_threads(1)
    meta, truth = load(run_dir); layers = meta['layers']
    cond = next(c for c in CONFIG['conditions'] if c['id']==cid)
    budget = 100*4**(layers-6) if budget_kind=='primary' else 100
    out = run_dir/f'raw/bbo/{cid}_{budget_kind}_seed{seed}.json'; assert not out.exists(), 'Preserve prior runs; do not overwrite.'
    d = {'status':'running','started_utc':now(),'seed':seed,'condition':cid,'budget_kind':budget_kind,'budget':budget}; save(out,d)
    try:
        if cond['kind']=='random': r = M.random_run(seed, layers, truth, budget)
        else: r = M.trajectory(seed, cond, layers, truth, budget, budget-40)
        d.update(r, status='completed', ended_utc=now()); save(out,d)
        print(cid, budget_kind, seed, 'evals', r['actual_evaluations'], 'best', r['final_best'], flush=True)
    except Exception:
        d.update(status='failed', traceback=traceback.format_exc(), ended_utc=now()); save(out,d); raise


def metrics(d, truth, cap):
    """Metrics restricted to the first `cap` evaluations of a finished run."""
    opt, worst = float(truth.min()), float(truth.max()); order = np.argsort(truth, kind='stable')
    rank = np.empty(len(truth), int); rank[order] = np.arange(1, len(truth)+1)
    ids = d['final_ids'][:cap]; ys = np.array(d['final_values'][:cap])
    hit = np.flatnonzero(np.abs(ys-opt) < 1e-12); best = float(ys.min())
    row = {'success': bool(len(hit)), 'first_hit': int(hit[0]+1) if len(hit) else cap+1, 'final_best': best,
           'regret': (best-opt)/(worst-opt), 'best_rank': int(rank[ids[int(np.argmin(ys))]]), 'evaluations': len(ids)}
    ev = [e for e in d.get('events', []) if e['evaluations_before'] < cap]
    if ev:
        row.update(cycles=len(ev), feasible=float(np.mean([e['raw_feasible_rate'] for e in ev])),
                   capture=float(np.mean([e['unseen_FM_min_hit'] for e in ev])),
                   no_candidate_cycles=int(sum(e['selected_id'] is None for e in ev)))
    return row


def analyze(run_dir):
    meta, truth = load(run_dir); layers = meta['layers']; primary = 100*4**(layers-6)
    # Views: 'primary' = full proportional budget; 'fixed100' = first 100 evaluations of the primary run
    # (Random has its own fixed100 run). Direct fixed100 runs of other conditions only serve the prefix check.
    rows, failed, checks = [], [], []
    for f in sorted((run_dir/'raw/bbo').glob('*.json')):
        d = json.loads(f.read_text())
        if d['status']!='completed': failed.append({'file':f.name,'status':d['status']}); continue
        cid = d['condition']['id']; seed = d['seed']
        if d['budget_kind']=='primary':
            rows.append({'condition':cid,'seed':seed,'budget_view':'primary','source':f.name,**metrics(d,truth,primary)})
            if primary>100 and cid!='Random':
                rows.append({'condition':cid,'seed':seed,'budget_view':'fixed100','source':f.name+'[:100]',**metrics(d,truth,100)})
        elif cid=='Random' or primary==100:
            rows.append({'condition':cid,'seed':seed,'budget_view':'fixed100','source':f.name,**metrics(d,truth,100)})
        else:
            p = run_dir/f'raw/bbo/{cid}_primary_seed{seed}.json'
            if p.exists():
                pp = json.loads(p.read_text())
                checks.append({'condition':cid,'seed':seed,'prefix_equal':pp['status']=='completed' and pp['final_ids'][:100]==d['final_ids']})
    groups = {}
    for r in rows:
        groups.setdefault((r['budget_view'],r['condition']),[]).append(r)
    def q(v): v=np.asarray(v,float); return [float(np.median(v)),float(np.percentile(v,25)),float(np.percentile(v,75))]
    summary = {}
    for (view,cid),rs in sorted(groups.items()):
        s = {'n':len(rs),'successes':sum(r['success'] for r in rs),'first_hit':q([r['first_hit'] for r in rs]),
             'best_rank':q([r['best_rank'] for r in rs]),'regret':q([r['regret'] for r in rs])}
        for k in ['feasible','capture','no_candidate_cycles']:
            if all(k in r for r in rs): s[k] = q([r[k] for r in rs])
        summary.setdefault(view,{})[cid] = s
    save(run_dir/'raw/summary.json', {'layers':layers,'candidates':len(truth),'primary_budget':primary,'generated_utc':now(),
         'summary':summary,'prefix_checks':checks,'unfinished':failed,'rows':rows})
    import csv
    with open(run_dir/'raw/per_run.csv','w',newline='') as fh:
        keys = sorted({k for r in rows for k in r}); w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)
    for view, s in summary.items():
        print(f'== L={layers} budget={view}')
        for cid, v in s.items():
            print(f"  {cid:9s} success {v['successes']}/{v['n']}  first_hit {v['first_hit'][0]:.0f} [{v['first_hit'][1]:.0f},{v['first_hit'][2]:.0f}]"
                  f"  best_rank {v['best_rank'][0]:.0f}  regret {v['regret'][0]:.4f}" + (f"  feas {v['feasible'][0]:.2f} cap {v['capture'][0]:.2f}" if 'feasible' in v else ''))
    print('prefix checks', checks)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['init','run','analyze']); ap.add_argument('--layers', type=int)
    ap.add_argument('--truth-from', type=Path); ap.add_argument('--run-dir', type=Path); ap.add_argument('--seed', type=int)
    ap.add_argument('--condition'); ap.add_argument('--budget', choices=['primary','fixed100'], default='primary')
    a = ap.parse_args()
    if a.command=='init': init(a.layers, a.truth_from)
    elif a.command=='run': run(a.run_dir.resolve(), a.seed, a.condition, a.budget)
    else: analyze(a.run_dir.resolve())
