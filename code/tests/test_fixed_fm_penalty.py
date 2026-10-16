"""Small SA smoke and fixed-model audit;does not run the1500-batch study."""
import copy
import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import evaluate_fixed_fm_penalty as e
from evaluator import CategoricalTableProblem
from qubo import build_onehot_penalty_bqm,qubo_to_ising_scale
from qarp_backend import all_bitstrings_lsb,one_hot_penalty,qubo_to_ising_coefficients
from sa_sampler import sample_sa


class FixedPenaltyTests(unittest.TestCase):
    def test_missing_candidates_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);labels=['a','b','c']
            (path/'schema.json').write_text(json.dumps({'parameters':[{'options':labels},{'options':labels}]}))
            with (path/'data.csv').open('w',newline='') as f:csv.writer(f).writerows([[a,b,float(3*i+j)] for i,a in enumerate(labels) for j,b in enumerate(labels)])
            p=CategoricalTableProblem(path/'data.csv',path/'schema.json');pred=np.arange(9,dtype=float)
            raw=np.array([p.patterns[0],p.patterns[1],p.patterns[1],np.zeros(6,dtype=np.int8)])
            d=e.measure(p,pred,[0],raw)
            self.assertEqual(d['selected_id'],1);self.assertEqual(d['fm_minimum_hit'],1)
            self.assertEqual(d['raw_feasible_rate'],.75);self.assertEqual(d['gap_score'],0.)
            for bits in [np.zeros((2,6),dtype=np.int8),np.array([p.patterns[0],p.patterns[0]])]:
                d=e.measure(p,pred,[0],bits);self.assertIsNone(d['selected_fm_gap']);self.assertEqual(d['fm_minimum_hit'],0)
                self.assertEqual(d['gap_score'],14.);self.assertIsNone(d['selected_id'])

    def test_small_energy_scale_and_sa_seed_metadata(self):
        for sizes in [(3,3),(3,3,3)]:
            n=sum(sizes);q=np.triu(np.random.default_rng(n).normal(0,.2,(n,n)));qd={(i,j):float(q[i,j]) for i in range(n) for j in range(i,n)}
            S=qubo_to_ising_scale(q);h,j=qubo_to_ising_coefficients(q)
            self.assertAlmostEqual(S,max(abs(v) for v in list(h.values())+list(j.values())))
            bits=all_bitstrings_lsb(n)
            for alpha in [.01,1.,100.]:
                bqm=build_onehot_penalty_bqm(qd,sizes,S*alpha,.3)
                np.testing.assert_allclose(bqm.energies((bits,list(range(n)))),np.einsum('bi,ij,bj->b',bits,q,bits)+.3+S*alpha*one_hot_penalty(bits,sizes),atol=1e-10,rtol=0)
                args=(qd,sizes,S*alpha,alpha,.3,{'num_reads':10,'num_sweeps':10},123)
                raw,meta=sample_sa(*args);again,_=sample_sa(*args);np.testing.assert_array_equal(raw,again)
                self.assertEqual(meta['sampler_seed'],123);self.assertEqual(len(meta['beta_range']),2)
                self.assertEqual(meta['penalty_lambda'],S*alpha)

    def test_frozen_source_resume_audit_and_figures(self):
        cfg=json.loads(e.DEFAULT_CONFIG.read_text());cfg=copy.deepcopy(cfg)
        cfg.update(seeds=[42],initialization_repeats=1,alphas=[.1,1.,100.],sampling_repeats=2)
        cfg['sa'].update(num_reads=10,num_sweeps=10)
        with tempfile.TemporaryDirectory() as tmp:
            cfg['output_dir']=tmp
            p,models,identity,out=e.plan(cfg);self.assertEqual(len(models),1)
            self.assertEqual(models[0]['info']['target_transform']['training_count'],40)
            e.run(cfg)
            with patch.object(e,'sample_sa',side_effect=AssertionError('No completed SA batches should rerun')):e.run(cfg)
            e.summarize(cfg)
            d=json.loads((out/'json/summary.json').read_text());self.assertEqual(d['audited_batches'],6);self.assertEqual(len(d['paired_tests']),6)
            for fmt in ['png','pdf','svg']:self.assertEqual(len(list((out/fmt).glob('*'))),2)
            shutil.copyfile(out/'png/penalty_accuracy.png','/private/tmp/fixed_penalty_smoke.png')
            path=out/'json/runs/seed42_init0.json';r=json.loads(path.read_text());r['batches'][0]['S']+=1;path.write_text(json.dumps(r))
            with self.assertRaises(AssertionError):e.summarize(cfg)


if __name__=='__main__':unittest.main()
