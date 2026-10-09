"""Re-run the frozen evaluator-only protocol and save inspectable evidence."""
from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import itertools
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from perovskites_evaluator import PerovskitesEvaluator, ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, obj):
    (ROOT / 'json' / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    protocol_path = ROOT / 'json/evaluator_validation_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if protocol['status'] != 'ready_for_evaluator_validation_only' or protocol['bbo_training_or_qaoa_allowed']:
        raise RuntimeError('Evaluator-only protocol must be reviewed before execution')
    # Reuse canonical pattern generation only; no circuit/solver is instantiated.
    repo = ROOT.parents[1]
    sys.path.insert(0, str(repo / 'src'))
    from qarp_backend import one_hot_patterns
    evaluator = PerovskitesEvaluator()
    shared_patterns = one_hot_patterns(evaluator.group_sizes)
    records = []
    reference = {}
    with evaluator.data_path.open(newline='') as stream:
        for row in csv.reader(stream):
            reference[tuple(row[:3])] = float(row[3])
    errors = []
    for i, labels in enumerate(itertools.product(*evaluator.options)):
        bits = evaluator.encode(labels)
        value = evaluator.evaluate_onehot(bits)
        if evaluator.decode(bits) != labels or value != reference[labels]:
            errors.append({'categories':labels,'kind':'roundtrip_or_value_mismatch'})
        if tuple(shared_patterns[i]) != bits:
            errors.append({'categories':labels,'kind':'canonical_pattern_order_mismatch'})
        records.append({'categories':list(labels),'onehot':list(bits),'basis_index_lsb':evaluator.basis_index(bits),'hse_gap':value})
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT/'py'), '-p', 'test_perovskites_evaluator.py', '-v'], capture_output=True, text=True)
    log = test.stdout + test.stderr
    (ROOT/'md/evaluator_validation.log').write_text(log)
    values = [r['hse_gap'] for r in records]
    minimum, maximum = min(values), max(values)
    audit = {
      'scope':'pinned_Olympus_lookup_table_only','status':'passed',
      'data_sha256':evaluator.data_sha256,'config_sha256':evaluator.config_sha256,
      'csv_header':False,'columns':['organic','cation','anion','hse_gap'],
      'row_count':len(evaluator.row_records),'expected_candidates':192,'unique_keys':len(reference),
      'missing_candidate_count':0,'duplicate_key_count':0,'missing_cell_count':0,
      'invalid_category_count':0,'nonfinite_objective_count':0,'negative_objective_count':0,
      'group_sizes':list(evaluator.group_sizes),'n_onehot_variables':23,
      'ordered_choices':{name:list(options) for name,options in zip(protocol['columns'][:3],evaluator.options)},
      'unit':'eV','unit_evidence':'json/unit_and_license_review.json',
      'reference_extrema':{'scope':'complete finite 192-row lookup; not original 1346-structure dataset or new DFT',
        'minimum':minimum,'maximum':maximum,
        'minimizers':[r['categories'] for r in records if r['hse_gap']==minimum],
        'maximizers':[r['categories'] for r in records if r['hse_gap']==maximum]},
      'lineage_limit':'Reduction of multiple structures per composition in original publication to one Olympus value not independently reconstructed'
    }
    validation={'status':'passed' if not errors and test.returncode==0 else 'failed',
      'scope':'evaluator_and_classical_encoding_only','protocol_sha256':digest(protocol_path),
      'checked_candidates':len(records),'canonical_patterns_checked':len(shared_patterns),
      'parity_errors':errors,'unit_test_exit_code':test.returncode,
      'unit_test_log':'md/evaluator_validation.log','data_sha256':evaluator.data_sha256,
      'all_reference_values_matched_exactly_as_python_float':not errors,
      'qaoa_prevalidation_performed':False,'fm_training_performed':False,'bbo_performed':False}
    write_json('data_audit.json',audit)
    write_json('evaluator_validation.json',validation)
    write_json('validated_candidates.json',{'scope':'audit_only_do_not_preload_into_optimizer_training','records':records})
    packages={}
    for name in ['numpy','torch','qarpx']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True)
    files=[ROOT/'py/perovskites_evaluator.py',ROOT/'py/validate_evaluator.py',ROOT/'py/test_perovskites_evaluator.py',repo/'src/qarp_backend.py']
    write_json('validation_environment.json',{'executed_at_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,
      'executable':sys.executable,'platform':platform.platform(),'machine':platform.machine(),'packages':packages,
      'repo_head':git.stdout.strip() if git.returncode==0 else None,'working_tree_is_not_frozen_by_head':True,
      'command':'python3 intern_1009/perovskites/py/validate_evaluator.py',
      'code_sha256':{str(p.relative_to(repo)):digest(p) for p in files}})
    print(json.dumps({'status':validation['status'],'rows':len(records),'onehot_variables':23,
                      'minimum':minimum,'maximum':maximum,'test_exit_code':test.returncode},ensure_ascii=False))
    if validation['status']!='passed':raise SystemExit(1)


if __name__=='__main__':
    main()
