"""Independent artifact checks after all declared seeds finish."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

OUT=Path(__file__).resolve().parents[1];PROBLEM=OUT.parent;REPO=PROBLEM.parents[1]
sys.path[:0]=[str(REPO/'src'),str(PROBLEM/'py')]
from perovskites_evaluator import PerovskitesEvaluator
from fm import TorchFM


def main():
    protocol_path=OUT/'json/protocol.json';protocol=json.loads(protocol_path.read_text());expected=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    env=json.loads((OUT/'json/environment.json').read_text())
    for path,digest in env['code_sha256'].items():assert hashlib.sha256((REPO/path).read_bytes()).hexdigest()==digest
    bb=PerovskitesEvaluator();assert bb.data_sha256==protocol['data_sha256']
    labels=list(bb.candidates());bits=np.array([bb.encode(c) for c in labels]);truth=np.array([bb.evaluate(c) for c in labels])
    checked=[]
    for seed in protocol['seeds']:
        row=json.loads((OUT/f'json/seed_{seed}.json').read_text());assert row['status']=='completed' and row['protocol_sha256']==expected
        initial=row['initial_dataset'];ids=initial['candidate_ids']
        assert len(ids)==20 and len(set(ids))==20
        assert ids==np.random.default_rng(seed).choice(192,20,replace=False).tolist()
        assert hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest()==row['initial_sha256']
        for record in initial['records']:assert record['hse_gap']==truth[record['candidate_id']]
        checkpoint=OUT/row['checkpoint'];assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==row['checkpoint_sha256']
        saved=torch.load(checkpoint,weights_only=True,map_location='cpu')
        h=hashlib.sha256()
        for key,value in saved['state_dict'].items():h.update(key.encode());h.update(value.numpy().tobytes())
        assert h.hexdigest()==row['fixed_model_sha256']
        model=TorchFM(23,2).cpu();model.load_state_dict(saved['state_dict']);model.eval()
        with torch.no_grad():pred=model(torch.tensor(bits,dtype=torch.float32)).numpy()
        q=np.array(row['qubo']);qpred=np.einsum('bi,ij,bj->b',bits,q,bits)+row['offset']
        assert np.max(np.abs(pred-qpred))<1e-4
        for name,m in row['methods'].items():
            assert m['initial_sha256']==row['initial_sha256'] and m['fixed_model_sha256']==row['fixed_model_sha256']
            assert len(m['raw_basis_indices'])==1000
            assert len(set(a['candidate_id'] for a in m['accepted_candidates']))==m['accepted_count']
            assert not set(a['candidate_id'] for a in m['accepted_candidates'])&set(ids)
            for a in m['accepted_candidates']:
                assert a['hse_gap']==truth[a['candidate_id']]
                assert abs(a['fm_prediction']-qpred[a['candidate_id']])<1e-10
            actual=min([truth[i] for i in ids]+[truth[a['candidate_id']] for a in m['accepted_candidates']])
            assert m['best_initial_plus_proposals']==actual
            assert abs(m['normalized_regret_initial_plus_proposals']-(actual-truth.min())/(truth.max()-truth.min()))<1e-12
            if name=='XY-FMQAOA':assert abs(1-m['exact_feasible_probability'])<1e-10 and m['lambda_internal']==0
            if name=='Penalty-FMQAOA':assert len(m['angle_search'])==9 and m['alpha']==5
            checked.append({'seed':seed,'method':name,'initial_and_model_match':True,'candidate_values_and_regret_match':True})
    (OUT/'json/artifact_verification.json').write_text(json.dumps({'status':'passed','checked_seed_method_records':checked,'source_and_checkpoint_hashes_verified':True},indent=2)+'\n')
    print('20 seed/method records and five FM checkpoints verified')


if __name__=='__main__':main()
