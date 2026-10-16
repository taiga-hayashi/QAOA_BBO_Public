"""Small smoke only;does not launch the195-fit study or change QAOA jobs."""
import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import evaluate_fm_settings as e
from fm_model import train_fm


class FMSettingsTests(unittest.TestCase):
    def test_split_and_prediction_only_candidate_selection(self):
        ids=list(range(24,64));truth=np.arange(64,dtype=float)
        train,unseen=e.data_split(ids,64)
        self.assertEqual(len(set(train)&set(unseen)),0)
        result=e.evaluate(truth.copy(),truth,ids)
        self.assertEqual(result['train_mse'],0)
        self.assertEqual(result['unseen_mse'],0)
        self.assertEqual(result['selected_id'],0)
        self.assertEqual(result['selected_true_regret'],0)
        self.assertEqual(result['top5_recall'],1)
        changed=truth.copy();changed[:24]=100-truth[:24]
        other=e.evaluate(truth.copy(),changed,ids)
        self.assertEqual(result['selected_id'],other['selected_id'])
        self.assertEqual(result['train_mse'],other['train_mse'])
        self.assertGreater(other['unseen_mse'],0)
        with self.assertRaises(ValueError):e.data_split([0]*40,64)
        with self.assertRaises(ValueError):e.evaluate(np.full(64,np.nan),truth,ids)

    def test_design_count_and_baseline(self):
        cfg=json.loads((e.CODE_DIR/'config_fm_training_fixed40.json').read_text())
        cases=e.variants(cfg);self.assertEqual(len(cases),13)
        for _,params,_,_ in cases[1:]:
            self.assertEqual(sum(params[k]!=v for k,v in cfg['baseline'].items()),1)
        self.assertEqual(len(cases)*len(cfg['seeds'])*cfg['initialization_repeats'],195)

    def test_smoke_train_resume_audit_and_render(self):
        cfg=json.loads((e.CODE_DIR/'config_fm_training_fixed40.json').read_text())
        cfg=copy.deepcopy(cfg);cfg['seeds']=[42];cfg['initialization_repeats']=1
        cfg['baseline']['epochs']=2;cfg['grid']={'rank':[1,2]}
        initial=json.loads(e.resolve(cfg['initial_sets_path']).read_text())['initial_ids']['42']
        p=e.load_problem(cfg['problem']);expected_x=p.patterns[initial].astype(np.float32)
        expected_y=[p.evaluate_id(i) for i in initial]
        def checked_train(x,y,d,params,seed):
            np.testing.assert_array_equal(x,expected_x)
            np.testing.assert_array_equal(y,expected_y)
            self.assertEqual(len(x),40)
            return train_fm(x,y,d,params,seed)
        with tempfile.TemporaryDirectory() as tmp:
            cfg['output_dir']=tmp
            with patch.object(e,'train_fm',side_effect=checked_train) as fit:e.run(cfg);self.assertEqual(fit.call_count,2)
            with patch.object(e,'train_fm',side_effect=AssertionError('completed fits must resume')):e.run(cfg)
            e.summarize(cfg)
            for fmt in ['png','pdf','svg']:
                files=list((Path(tmp)/fmt).glob('*'));self.assertEqual(len(files),1);self.assertGreater(files[0].stat().st_size,1000)
            shutil.copyfile(Path(tmp)/'png/fm_rank.png','/private/tmp/fm_settings_smoke.png')
            rec=Path(tmp)/'json/runs/baseline_seed42_init0.json'
            d=json.loads(rec.read_text());d['metrics']['selected_true_regret']+=.1;rec.write_text(json.dumps(d))
            with self.assertRaises(AssertionError):e.summarize(cfg)


if __name__=='__main__':unittest.main()
