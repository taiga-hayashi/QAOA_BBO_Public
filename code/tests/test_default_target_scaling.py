"""Optical defaults exercise the real training-to-sampler path in original units."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import main
from config import CODE_DIR,load_config
from evaluator import load_problem
from fm_target_scaling import predict_original
from qubo import qubo_to_ising_scale


class DefaultScalingTests(unittest.TestCase):
    def test_optical_defaults_and_legacy_problem_scope(self):
        cfg=load_config(['--config',str(CODE_DIR/'config_multilayer_filter.json')])
        self.assertEqual(cfg['fm'],dict(rank=4,epochs=120,learning_rate=.1,optimizer='AdamW',weight_decay=.1,target_standardization=True))
        legacy=load_config(['--config',str(CODE_DIR/'config.json')])
        self.assertEqual(legacy['problem']['name'],'perovskites')
        self.assertFalse(legacy['fm'].get('target_standardization',False))

    def test_original_units_every_method_and_checkpoint_resume(self):
        cfg=load_config(['--config',str(CODE_DIR/'config_multilayer_filter.json')]);cfg=copy.deepcopy(cfg)
        cfg['methods']=['Exact-FM','Adaptive-FMQA','LargePenalty-FMQA','XY-FMQAOA']
        cfg['problem']['bbo_cycles']=2;cfg['sa'].update(num_reads=10,num_sweeps=10)
        cfg['qaoa']['angle_search_mode']='legacy9'
        p=load_problem(cfg['problem']);fit=main.train_scaled_fm;dispatch=main.run_single_step_sampler
        state={};calls=[]
        def checked_train(x,y,d,params,seed,standardize):
            self.assertTrue(standardize)
            model,transform=fit(x,y,d,params,seed,standardize)
            self.assertAlmostEqual(transform['mean'],float(np.mean(y)))
            self.assertAlmostEqual(transform['scale'],float(np.std(y)))
            self.assertEqual(transform['training_count'],len(x))
            state.update(model=model,transform=transform);return model,transform
        def checked_sampler(**kwargs):
            q=kwargs['q_mat'];offset=kwargs['offset'];scale=kwargs['ising_scale']
            expected=predict_original(state['model'],p.patterns,state['transform'])
            actual=np.einsum('bi,ij,bj->b',p.patterns,q,p.patterns)+offset
            np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=0)
            for (i,j),value in kwargs['q_dict'].items():self.assertAlmostEqual(value,q[i,j])
            self.assertAlmostEqual(scale,qubo_to_ising_scale(q))
            bits,metadata=dispatch(**kwargs)
            np.testing.assert_allclose(metadata['penalty_lambda'],metadata['penalty_alpha']*scale,atol=1e-10)
            calls.append(kwargs['method']);return bits,metadata
        with tempfile.TemporaryDirectory() as tmp:
            cp=Path(tmp)/'json/seed_42.json';cp.parent.mkdir()
            with patch.object(main,'train_scaled_fm',side_effect=checked_train),patch.object(main,'run_single_step_sampler',side_effect=checked_sampler):
                full=main.run_seed(42,p,cfg,checkpoint_path=cp)
            self.assertEqual(len(calls),8)
            self.assertEqual(len({r['step0_fm_sha256'] for r in full['methods'].values()}),1)
            for record in full['methods'].values():self.assertTrue(all(h['target_transform']['standardized'] for h in record['history']))
            checkpoint=torch.load(Path(tmp)/'checkpoints/seed42_method0_latest.pt',weights_only=True)
            self.assertTrue(checkpoint['target_transform']['standardized'])
            # Simulate interruption after the first adaptive proposal.
            partial=copy.deepcopy(full);partial['methods']={m:copy.deepcopy(full['methods'][m]) for m in cfg['methods'][:2]}
            previous=partial['methods']['Adaptive-FMQA'];previous.update(status='running',history=previous['history'][:1],total_elapsed_sec=0.)
            resumed=main.run_seed(42,p,cfg,existing_rec=partial)
            for method in cfg['methods']:
                self.assertEqual([h['chosen_candidate_id'] for h in full['methods'][method]['history']],
                                 [h['chosen_candidate_id'] for h in resumed['methods'][method]['history']])


if __name__=='__main__':unittest.main()
