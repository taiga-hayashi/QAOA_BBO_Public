"""mlfilter-scale Phase 0: feasibility measurements for L-layer filters (diagnostic only).

Subcommands (one run directory per L):
  init      --layers L               create data/runs/mlfilter-scale/<config_id>/<run_id>/
  enumerate --run-dir D              exhaustive truth table + independent re-evaluation + scalar TMM check
  timing    --run-dir D              FM / SA / compact-XY timings on one seed (needs a passed enumeration)
  xstate    --run-dir D              one standard-QAOA statevector evaluation (time, peak memory)

No method comparison is made here. The truth table is used only to give the FM
its 40 charged training values and to report sizes; samplers never read it.
"""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import random
import resource
import secrets
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

STUDY_DIR = Path(__file__).resolve().parents[1]
PROJECT = STUDY_DIR.parents[1]
CONFIG = json.loads((STUDY_DIR/'config.json').read_text())
PATHS = json.loads((PROJECT/'config/paths.local.json').read_text())['profiles']['server']
INTERN = Path(PATHS['intern_fujitsu_root'])
sys.path[:0] = [str(Path(__file__).resolve().parent), str(PROJECT/'src'), str(INTERN/'src'), str(INTERN), PATHS['extra_pythonpath']]
from evaluator import FilterEvaluator, sha, save

DEPENDENCIES = [INTERN/p for p in ['src/fm.py','src/fm_to_qubo.py','src/fmqa_solver.py','src/bb_function.py',
                                    'src/qarp_backend.py','intern_0924/src/solvers.py','intern_0924/src/problem.py',
                                    'intern_0924/src/penalty.py']] + [PROJECT/'src/qarp_pair_xy.py']


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def peak_rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20


def code_hashes():
    paths = [Path(__file__).resolve(), STUDY_DIR/'py/evaluator.py', STUDY_DIR/'config.json', *DEPENDENCIES]
    return {str(p):sha(p) for p in paths}


def environment():
    names = ['numpy','scipy','torch','dimod','dwave-neal','openqarp']
    packages = {}
    for name in names:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    return {'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'hostname':socket.gethostname(),'packages':packages,'threads':CONFIG['threads']}


def labels_for(layers):
    return np.array(list(itertools.product(range(4),repeat=layers)),dtype=np.int8)


def matrix(qdict,n):
    # Same as intern_1009/perovskites/fixed_fm_pilot/py/run_pilot.py:matrix (upper triangle).
    q = np.zeros((n,n))
    for (i,j),v in qdict.items(): q[min(i,j),max(i,j)] += v
    return q


def load_meta(run_dir):
    meta = json.loads((run_dir/'metadata.json').read_text())
    assert meta['code_sha256'] == code_hashes(), 'code changed since init; create a new run'
    return meta


# ---------------------------------------------------------------- init
def init(layers, truth_from=None):
    condition = {'config':CONFIG,'layers':layers}
    config_hash = canonical_sha(condition)
    config_id = 'c-'+config_hash[:12]
    base = PROJECT/'data/runs/mlfilter-scale'/config_id
    while True:
        run_id = 'r-'+secrets.token_hex(6)
        if not (base/run_id).exists(): break
    run_dir = base/run_id
    (run_dir/'raw').mkdir(parents=True); (run_dir/'logs').mkdir()
    save(run_dir/'metadata.json',{'study':'mlfilter-scale','phase':'phase0','protocol_phase':'diagnostic','layers':layers,
         'candidates':4**layers,'n_onehot_variables':4*layers,'config_hash':config_hash,'config_id':config_id,
         'config_hash_method':'sha256 of canonical JSON {config: experiments/mlfilter-scale/config.json, layers: L}',
         'run_id':run_id,'created_utc':now(),'code_sha256':code_hashes(),'environment':environment(),
         'truth_from':truth_source(truth_from, layers) if truth_from else None})
    save(run_dir/'status.json',{'compute_status':'pending','validation_status':'pending','package_validation_status':'pending',
         'storage_status':'pending','analysis_status':'not_required','overall_status':'in_progress','updated_at_utc':now()})
    print(run_dir)


def truth_source(run_dir, layers):
    # Reuse a passed enumeration from an earlier run of the same L (checked again at use).
    run_dir = Path(run_dir).resolve(); d = json.loads((run_dir/'raw/enumeration.json').read_text())
    assert d['status']=='completed' and d['validation']=='passed' and d['layers']==layers
    assert sha(run_dir/'raw/truth_table.npz')==d['truth_table_sha256']
    return {'run_dir':str(run_dir.relative_to(PROJECT)),'enumeration_sha256':sha(run_dir/'raw/enumeration.json'),
            'truth_table_sha256':d['truth_table_sha256']}


def set_status(run_dir, **kw):
    path = run_dir/'status.json'; d = json.loads(path.read_text()); d.update(kw, updated_at_utc=now()); save(path,d)


# ---------------------------------------------------------------- enumerate
def evaluate_all(evaluator, labels, batch, progress=None):
    values = np.empty(len(labels)); start = time.perf_counter()
    for first in range(0,len(labels),batch):
        values[first:first+batch] = evaluator.evaluate(labels[first:first+batch])
        if progress and (first//batch) % max(1,4096//batch) == 0:
            progress(first+batch, time.perf_counter()-start)
    return values, time.perf_counter()-start


def enumerate_all(run_dir):
    meta = load_meta(run_dir); layers = meta['layers']; cfg = CONFIG['enumeration']
    out = run_dir/'raw/enumeration.json'; table = run_dir/'raw/truth_table.npz'
    assert not out.exists() and not table.exists(), 'Preserve prior runs; do not overwrite.'
    d = {'status':'running','started_utc':now(),'layers':layers,'candidates':4**layers}; save(out,d)
    try:
        e = FilterEvaluator(layers); labels = labels_for(layers)
        np.testing.assert_array_equal(e.decode(e.encode(labels[:4096])),labels[:4096])
        def report(done,sec):
            d.update(primary_done=min(done,len(labels)),primary_seconds_so_far=sec); save(out,d)
            print(f'primary {min(done,len(labels))}/{len(labels)} {sec:.0f}s',flush=True)
        fom, sec1 = evaluate_all(e, labels, cfg['batch_primary'], report)
        assert np.isfinite(fom).all() and (fom>=0).all()
        np.savez(table, labels=labels, fom=fom)
        d.update(primary_seconds=sec1, truth_table='raw/truth_table.npz', truth_table_sha256=sha(table)); save(out,d)
        print('independent pass',flush=True)
        check, sec2 = evaluate_all(e, labels, cfg['batch_independent'])
        diff = float(np.max(abs(check-fom)))
        best = int(np.argmin(fom))
        sys.path.insert(0, str(PROJECT/'data/inputs/mlfilter/v1-public-code/vendor/site-packages'))
        from tmm import coh_tmm as scalar_tmm
        t_best = e.spectra(labels[best:best+1])[0][0]
        scalar = np.array([scalar_tmm('p',[1.4,*e.index[labels[best],w].tolist(),1.45],[np.inf,*([50e-9]*layers),np.inf],0.,
                                      e.wavelength[w])['T'] for w in range(2201)])
        scalar_fom = float(e.fom(scalar[None,:])[0])
        d.update(independent_seconds=sec2, independent_batch=cfg['batch_independent'], independent_max_abs_difference=diff,
                 best_index=best, best_labels=labels[best].tolist(), best_materials=[e.materials[i] for i in labels[best]],
                 min_fom=float(fom[best]), max_fom=float(fom.max()), median_fom=float(np.median(fom)),
                 second_best_gap=float(np.partition(fom,1)[1]-fom[best]),
                 scalar_tmm_best_max_T_error=float(np.max(abs(scalar-t_best))), scalar_tmm_best_fom=scalar_fom,
                 scalar_tmm_best_fom_error=abs(scalar_fom-float(fom[best])))
        if layers == 6:
            ref = INTERN/'intern1016/multilayer_filter/data/lookup_table.npz'
            r = np.load(ref); np.testing.assert_array_equal(r['labels'],labels)
            d.update(reference_table=str(ref), reference_table_sha256=sha(ref),
                     reference_max_abs_difference=float(np.max(abs(r['fom']-fom))),
                     reference_best_index_match=int(np.argmin(r['fom']))==best)
        passed = diff <= cfg['max_abs_difference_tolerance'] and d['scalar_tmm_best_fom_error'] <= 1e-10 \
                 and d['scalar_tmm_best_max_T_error'] <= 1e-10 and d.get('reference_max_abs_difference',0.) <= 1e-11
        d.update(status='completed', validation='passed' if passed else 'failed', ended_utc=now(), peak_rss_gib=peak_rss_gib())
        save(out,d); set_status(run_dir, enumeration='completed', enumeration_validation=d['validation'])
        print('enumeration', d['validation'], f'min {d["min_fom"]:.11f}', d['best_materials'], flush=True)
    except Exception:
        d.update(status='failed', traceback=traceback.format_exc(), ended_utc=now()); save(out,d)
        set_status(run_dir, enumeration='failed'); raise


def require_enumeration(run_dir):
    src = json.loads((run_dir/'metadata.json').read_text()).get('truth_from')
    if src:
        run_dir = PROJECT/src['run_dir']; assert sha(run_dir/'raw/enumeration.json')==src['enumeration_sha256']
    d = json.loads((run_dir/'raw/enumeration.json').read_text())
    assert d['status']=='completed' and d['validation']=='passed'
    table = run_dir/'raw/truth_table.npz'; assert sha(table)==d['truth_table_sha256']
    return d, np.load(table)


# ---------------------------------------------------------------- timing
class BudgetReached(Exception):
    pass


def fit_fm(layers, labels, fom, seed):
    import torch
    from fm import train_factorization_machine
    from fm_to_qubo import fm_to_qubo
    from qarp_backend import qubo_to_ising_coefficients
    n = 4*layers; cfg = CONFIG['timing']['training']
    ids = random.Random(seed).sample(range(4**layers), CONFIG['timing']['initial_points'])
    x = np.eye(4,dtype=np.float32)[labels[ids]].reshape(len(ids),n)
    torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    t = time.perf_counter()
    model = train_factorization_machine(torch.tensor(x), torch.tensor(fom[ids],dtype=torch.float32), n, k=cfg['rank'],
        epochs=cfg['epochs'], learning_rate=cfg['learning_rate'], optimizer_name=cfg['optimizer'], weight_decay=cfg['weight_decay'])
    train_sec = time.perf_counter()-t; model.eval()
    qd, b = fm_to_qubo(model); q = matrix(qd, n)
    h, j = qubo_to_ising_coefficients(q)
    scale = max([abs(v) for v in h.values()]+[abs(v) for v in j.values()]+[0.]); assert scale > 0
    return model, ids, qd, q, b, scale, train_sec


def predictions(q, b, labels, layers, chunk=1<<16):
    pred = np.empty(len(labels)); eye = np.eye(4)
    for first in range(0,len(labels),chunk):
        x = eye[labels[first:first+chunk]].reshape(-1,4*layers)
        pred[first:first+chunk] = np.einsum('bi,ij,bj->b',x,q,x)+b
    return pred


def xy_search_1024(backend, pred, seed):
    """Same budget layout as experiments/mlfilter-xydepth search() at depth 1, Mean objective."""
    import itertools as it
    from scipy.optimize import minimize
    from qarp_backend import OpenQARPCompactXYQAOA
    grid = json.loads((INTERN/'intern1016/multilayer_filter/fixed_fm/json/protocol.json').read_text())['angles']
    depth = 1; trace = []; maxima = np.array([2*np.pi]*depth+[np.pi]*depth); bounds = [(0.,v) for v in maxima]
    rng = np.random.default_rng(np.random.SeedSequence([seed,0,1016,49,depth]))
    def evaluate(a, stop=1024, phase='global'):
        if len(trace) >= stop: raise BudgetReached()
        if len(trace) % 32 == 0: OpenQARPCompactXYQAOA._mixer_cache.clear()
        a = np.asarray(a); prob = backend.feasible_probabilities(a[:depth],a[depth:])
        value = float(prob@pred); trace.append({'angles':a.tolist(),'phase':phase,'mean':value}); return value
    for gamma,beta in it.product(grid['gammas'],grid['betas']): evaluate([gamma,beta],phase='p1_grid49')
    while len(trace) < 256: evaluate(rng.random(2*depth)*maxima,phase='global_to256')
    for a in [r['angles'] for r in sorted(trace,key=lambda r:r['mean'])[:4]]:
        stop = len(trace)+192
        try: minimize(lambda v: evaluate(v,stop,'Powell'), a, method='Powell', bounds=bounds,
                      options={'maxfev':192,'xtol':1e-6,'ftol':1e-9})
        except BudgetReached: pass
    while len(trace) < 1024: evaluate(rng.random(2*depth)*maxima,phase='remaining_global')
    assert len(trace) == 1024
    return min(trace,key=lambda r:r['mean'])


def timing(run_dir):
    import torch, neal
    from fmqa_solver import build_penalty_bqm
    from qarp_backend import OpenQARPCompactXYQAOA, OpenQARPXYQAOA
    from qarp_pair_xy import OpenQARPPairXYQAOA
    torch.set_num_threads(1)
    meta = load_meta(run_dir); layers = meta['layers']; cfg = CONFIG['timing']; sizes = [4]*layers; n = 4*layers
    out = run_dir/'raw/timing.json'; assert not out.exists(), 'Preserve prior runs; do not overwrite.'
    enum, table = require_enumeration(run_dir); labels, fom = table['labels'], table['fom']
    d = {'status':'running','started_utc':now(),'layers':layers,'seed':cfg['model_seed']}; save(out,d)
    try:
        seed = cfg['model_seed']
        model, ids, qd, q, b, scale, train_sec = fit_fm(layers, labels, fom, seed)
        t = time.perf_counter(); pred = predictions(q, b, labels, layers); pred_sec = time.perf_counter()-t
        unseen = np.setdiff1d(np.arange(len(labels)), ids)
        d.update(initial_ids=ids, fm_train_seconds=train_sec, fm_predict_all_seconds=pred_sec, S=scale,
                 fm_unseen_min_index=int(unseen[np.argmin(pred[unseen])]), truth_rank_of_fm_unseen_min=int(np.sum(fom < fom[unseen[np.argmin(pred[unseen])]])),
                 peak_rss_gib_after_fm=peak_rss_gib()); save(out,d)
        print(f'FM {train_sec:.2f}s predict {pred_sec:.2f}s',flush=True)
        # SA: one batch (alpha=1), the sampler does not see the truth table.
        lam = scale*cfg['sa']['alpha']; t = time.perf_counter()
        result = neal.SimulatedAnnealingSampler().sample(build_penalty_bqm(qd,sizes,lam,b), num_reads=cfg['sa']['num_reads'],
                     num_sweeps=cfg['sa']['num_sweeps'], seed=seed*100000, beta_schedule_type=cfg['sa']['beta_schedule_type'])
        sa_sec = time.perf_counter()-t
        arr = result.record.sample[:,[list(result.variables).index(i) for i in range(n)]]
        raw = np.repeat(arr,result.record.num_occurrences,axis=0).reshape(-1,layers,4)
        d.update(sa_batch_seconds=sa_sec, sa_feasible_rate=float((raw.sum(2)==1).all(1).mean()), sa_lambda=lam); save(out,d)
        print(f'SA {sa_sec:.2f}s',flush=True)
        # Compact XY: build, single-angle repeats, crosscheck, then one 1024-evaluation search.
        t = time.perf_counter(); backend = OpenQARPPairXYQAOA(q/scale, sizes); build_sec = time.perf_counter()-t
        singles = []
        for rep in range(cfg['xy']['single_angle_repeats']):
            OpenQARPCompactXYQAOA._mixer_cache.clear(); t = time.perf_counter()
            prob = backend.feasible_probabilities([.31],[.27]); singles.append(time.perf_counter()-t)
        assert abs(prob.sum()-1) < 1e-10
        d.update(xy_backend='OpenQARPPairXYQAOA', xy_build_seconds=build_sec, xy_single_angle_seconds=singles,
                 xy_simulator_qubits=backend.simulator_qubits, xy_feasible_states=len(prob), peak_rss_gib_after_xy_single=peak_rss_gib())
        if layers in cfg['xy']['compact_crosscheck_layers']:
            errs = []
            for a in [[.31,.27],[1.7,.9],[4.2,2.5]]:
                ref = OpenQARPCompactXYQAOA(q/scale, sizes).feasible_probabilities([a[0]],[a[1]])
                errs.append(float(np.max(abs(ref-backend.feasible_probabilities([a[0]],[a[1]])))))
            d.update(pair_vs_compact_max_abs_error=max(errs)); assert max(errs) < cfg['xy']['compact_crosscheck_tolerance']
        if layers in cfg['xy']['full_xy_crosscheck_layers']:
            t = time.perf_counter(); full = OpenQARPXYQAOA(q/scale, sizes); state = full.statevector([.31],[.27])
            err = float(np.max(abs(prob-abs(state[full.feasible_indices])**2)))
            d.update(full_xy_qubits=n, full_xy_seconds=time.perf_counter()-t, full_vs_compact_max_abs_error=err,
                     full_xy_feasible_mass=float(np.sum(abs(state[full.feasible_indices])**2)))
            del full, state
            assert err < 1e-10
        save(out,d); print(f'XY single {np.median(singles):.3f}s; 1024 search',flush=True)
        t = time.perf_counter(); best = xy_search_1024(backend, pred, seed); search_sec = time.perf_counter()-t
        d.update(xy_search_evaluations=1024, xy_search_seconds=search_sec, xy_search_best=best,
                 status='completed', ended_utc=now(), peak_rss_gib=peak_rss_gib())
        save(out,d); set_status(run_dir, timing='completed'); print(f'XY 1024 search {search_sec:.0f}s',flush=True)
    except Exception:
        d.update(status='failed', traceback=traceback.format_exc(), ended_utc=now(), peak_rss_gib=peak_rss_gib()); save(out,d)
        set_status(run_dir, timing='failed'); raise


# ---------------------------------------------------------------- xstate
def xstate(run_dir):
    import torch
    from fmqa_solver import build_penalty_bqm
    from qarp_backend import OpenQARPStandardQAOA
    from intern_0924.src.solvers import _build_qubo_energies, _state_energy_expectation
    torch.set_num_threads(1)
    meta = load_meta(run_dir); layers = meta['layers']; cfg = CONFIG['timing']; sizes = [4]*layers; n = 4*layers
    out = run_dir/'raw/xstate.json'; assert not out.exists(), 'Preserve prior runs; do not overwrite.'
    enum, table = require_enumeration(run_dir); labels, fom = table['labels'], table['fom']
    d = {'status':'running','started_utc':now(),'layers':layers,'qubits':n,
         'statevector_gib':(1<<n)*16/2**30,'energy_table_gib':(1<<n)*8/2**30}; save(out,d)
    try:
        model, ids, qd, q, b, scale, _ = fit_fm(layers, labels, fom, cfg['model_seed'])
        alpha = cfg['standard_qaoa']['alpha']; gamma, beta = cfg['standard_qaoa']['angle']
        t = time.perf_counter(); backend = OpenQARPStandardQAOA(q/scale, sizes, alpha); state = backend.statevector([gamma],[beta])
        state_sec = time.perf_counter()-t; d.update(statevector_seconds=state_sec, peak_rss_gib_after_state=peak_rss_gib()); save(out,d)
        print(f'state {state_sec:.1f}s peak {peak_rss_gib():.1f} GiB',flush=True)
        bqm = build_penalty_bqm({k:v/scale for k,v in qd.items()}, sizes, alpha, b/scale); total, offset = bqm.to_qubo()
        t = time.perf_counter(); energies = _build_qubo_energies(matrix(total,n), offset, np.dtype(np.float64)); energy_sec = time.perf_counter()-t
        t = time.perf_counter(); value, norm = _state_energy_expectation(state, energies, 1<<16); expect_sec = time.perf_counter()-t
        assert abs(norm-1) < 1e-10
        valid = np.eye(4,dtype=np.uint64)[labels].reshape(-1,n) @ (np.uint64(1) << np.arange(n,dtype=np.uint64))
        d.update(energy_table_seconds=energy_sec, expectation_seconds=expect_sec, normalized_objective=value, norm=norm,
                 feasible_probability=float(np.sum(abs(state[valid])**2)), status='completed', ended_utc=now(), peak_rss_gib=peak_rss_gib())
        save(out,d); set_status(run_dir, xstate='completed'); print(f'energies {energy_sec:.1f}s expect {expect_sec:.1f}s',flush=True)
    except Exception:
        d.update(status='failed', traceback=traceback.format_exc(), ended_utc=now(), peak_rss_gib=peak_rss_gib()); save(out,d)
        set_status(run_dir, xstate='failed'); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['init','enumerate','timing','xstate'])
    parser.add_argument('--layers', type=int); parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--truth-from', type=Path, help='init: reuse the passed truth table of an earlier run (same L)')
    args = parser.parse_args()
    if args.command == 'init': init(args.layers, args.truth_from)
    else: {'enumerate':enumerate_all,'timing':timing,'xstate':xstate}[args.command](args.run_dir.resolve())
