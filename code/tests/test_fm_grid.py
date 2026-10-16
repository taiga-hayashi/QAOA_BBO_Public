"""Scaling leakage,full-factorial design,raw-scale predictions and artifact audits."""
import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import evaluate_fm_grid as grid
import fm_target_scaling as scaling
from fm_model import train_fm,model_sha256
from qubo import fm_to_qubo


class GridTests(unittest.TestCase):
    def test_training_only_scaling_constant_and_raw_parity(self):
        x=np.eye(4,dtype=np.float32);y=np.array([1.,2.,3.,4.])
        normalized,t=scaling.fit_target_transform(y,True)
        self.assertAlmostEqual(float(normalized.mean()),0.)
        self.assertAlmostEqual(float(normalized.std()),1.)
        np.testing.assert_allclose(normalized*t['scale']+t['mean'],y,atol=1e-15)
        constant,t0=scaling.fit_target_transform([3.,3.],True)
        np.testing.assert_array_equal(constant,[0.,0.]);self.assertEqual(t0['scale'],1.)
        params={'rank':2,'epochs':3,'learning_rate':.1,'optimizer':'AdamW','weight_decay':.01}
        plain=train_fm(x,y,4,params,42)
        raw,tr=scaling.train_scaled_fm(x,y,4,params,42,False)
        self.assertEqual(model_sha256(plain),model_sha256(raw));self.assertEqual(tr['scale'],1.)
        real=scaling.train_fm
        def checked(x0,y0,d,p,seed):
            np.testing.assert_array_equal(x0,x);np.testing.assert_allclose(y0,normalized)
            return real(x0,y0,d,p,seed)
        with patch.object(scaling,'train_fm',side_effect=checked):model,transform=scaling.train_scaled_fm(x,y,4,params,42,True)
        _,q,b=fm_to_qubo(model);qr,br=scaling.qubo_original_scale(q,b,transform)
        np.testing.assert_allclose(np.einsum('bi,ij,bj->b',x,qr,x)+br,scaling.predict_original(model,x,transform),atol=1e-5,rtol=0)

    def test_full_grid_and_tests_declared(self):
        cfg=json.loads(grid.DEFAULT_CONFIG.read_text())
        self.assertEqual(len(grid.variants(cfg)),128)
        self.assertEqual(len(grid.variants(cfg))*len(cfg['seeds'])*cfg['initialization_repeats'],1920)
        self.assertEqual(len(grid.contrasts(cfg))*len(cfg['primary_metrics']),573)
        self.assertEqual(sum(std for _,_,std in grid.variants(cfg)),64)
        self.assertTrue(all(p['epochs']==120 for _,p,_ in grid.variants(cfg)))

    def test_end_to_end_inverse_predictions_and_resume(self):
        cfg=json.loads(grid.DEFAULT_CONFIG.read_text());cfg=copy.deepcopy(cfg)
        cfg['seeds']=[42];cfg['initialization_repeats']=1;cfg['baseline']['epochs']=2
        cfg['grid']={'rank':[1,2],'learning_rate':[.1],'weight_decay':[.01,.1]}
        with tempfile.TemporaryDirectory() as tmp:
            cfg['output_dir']=tmp
            p,initial,_,_=grid.setup(cfg);ids=initial['initial_ids']['42']
            expected_x=p.patterns[ids].astype(np.float32);expected_y=np.array([p.evaluate_id(i) for i in ids])
            real=grid.train_scaled_fm
            def checked(x,y,d,params,seed,std):
                np.testing.assert_array_equal(x,expected_x);np.testing.assert_array_equal(y,expected_y)
                model,t=real(x,y,d,params,seed,std)
                self.assertEqual(t['training_count'],40)
                self.assertEqual(t['mean'],float(expected_y.mean()) if std else 0.)
                return model,t
            with patch.object(grid,'train_scaled_fm',side_effect=checked) as fit:grid.run(cfg);self.assertEqual(fit.call_count,8)
            with patch.object(grid,'train_scaled_fm',side_effect=AssertionError('Must skip completed fits')):grid.run(cfg)
            grid.summarize(cfg)
            summary=json.loads((Path(tmp)/'json/summary.json').read_text());self.assertEqual(summary['fit_count'],8)
            for fmt in ['png','pdf','svg']:self.assertEqual(len(list((Path(tmp)/fmt).glob('*'))),4)
            shutil.copyfile(Path(tmp)/'png/grid_unseen_rmse.png','/private/tmp/fm_grid_smoke.png')
            path=next((Path(tmp)/'json/runs').glob('*.json'));r=json.loads(path.read_text());r['target_transform']['mean']+=1;path.write_text(json.dumps(r))
            with self.assertRaises(AssertionError):grid.read_and_audit(cfg)


if __name__=='__main__':unittest.main()
