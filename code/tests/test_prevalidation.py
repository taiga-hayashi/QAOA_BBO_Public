"""Independent small-circuit checks and interruption/restart contract tests."""
import copy
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_config
from evaluator import load_problem
from fm_model import train_fm, model_sha256
from qubo import fm_to_qubo, qubo_to_ising_scale, build_onehot_penalty_bqm
from qarp_backend import (OpenQARPXYQAOA, OpenQARPCompactXYQAOA,
                          OpenQARPStandardQAOA, all_bitstrings_lsb,
                          one_hot_penalty, qubo_to_ising_coefficients)
from shared_numerics import _build_qubo_energies, AdaptivePenaltyTracker
import main

CODE = Path(__file__).resolve().parents[1]


def independent_state(q, sizes, gamma, beta, xy):
    n = len(q)
    bits = all_bitstrings_lsb(n)
    if xy:
        feasible = one_hot_penalty(bits, sizes) == 0
        state = feasible.astype(complex) / np.sqrt(feasible.sum())
    else:
        state = np.ones(1 << n, complex) / np.sqrt(1 << n)
    energy = np.einsum('bi,ij,bj->b', bits, q, bits)
    for gamma,beta in zip(np.atleast_1d(gamma),np.atleast_1d(beta)):
        state *= np.exp(-1j * gamma * energy)
        start = 0
        if xy:
            for size in sizes:
                edges = [(start+i, start+(i+1) % size) for i in range(size)] if size > 2 else [(start,start+1)]
                for a, b in edges:
                    left = [i for i in range(1 << n) if ((i>>a)&1)==0 and ((i>>b)&1)==1]
                    right = [i ^ (1<<a) ^ (1<<b) for i in left]
                    u,v=state[left].copy(),state[right].copy()
                    state[left]=np.cos(beta)*u-1j*np.sin(beta)*v
                    state[right]=-1j*np.sin(beta)*u+np.cos(beta)*v
                start += size
        else:
            for a in range(n):
                left = [i for i in range(1 << n) if ((i>>a)&1)==0]
                right = [i ^ (1<<a) for i in left]
                u,v=state[left].copy(),state[right].copy()
                state[left]=np.cos(beta)*u-1j*np.sin(beta)*v
                state[right]=-1j*np.sin(beta)*u+np.cos(beta)*v
    return state


class Prevalidation(unittest.TestCase):
    def test_circuits_energy_penalty_and_bit_order(self):
        for sizes in [(3,3),(3,3,3),(4,4)]:
            n=sum(sizes);q=np.triu(np.random.default_rng(n).normal(0,.2,(n,n)))
            bits=all_bitstrings_lsb(n);pen=one_hot_penalty(bits,sizes)
            self.assertTrue(np.all(pen[pen!=0]>0))
            backend=OpenQARPXYQAOA(q,sizes)
            compact=OpenQARPCompactXYQAOA(q,sizes)
            self.assertEqual(backend.onehot_penalty_lambda,0)
            np.testing.assert_array_equal(bits[backend.feasible_indices],backend.feasible_patterns)
            for gamma,beta in [(0.,0.),(.31,.27),(2.1,1.3)]:
                sv=backend.statevector([gamma],[beta])
                self.assertLess(abs(1-np.sum(abs(sv[backend.feasible_indices])**2)),1e-10)
                ref=independent_state(q,sizes,gamma,beta,True)
                np.testing.assert_allclose(abs(sv)**2,abs(ref)**2,atol=1e-10,rtol=0)
                np.testing.assert_allclose(compact.feasible_probabilities([gamma],[beta]),abs(ref[backend.feasible_indices])**2,atol=1e-10,rtol=0)
            for alpha in [0.,5.,100.]:
                qd={(i,j):q[i,j] for i in range(n) for j in range(i,n)}
                bqm=build_onehot_penalty_bqm(qd,sizes,alpha,.3)
                total,offset=bqm.to_qubo();mat=np.zeros_like(q)
                for (i,j),v in total.items():mat[min(i,j),max(i,j)]+=v
                energy=_build_qubo_energies(mat,offset,np.dtype(float))
                target=np.einsum('bi,ij,bj->b',bits,q,bits)+.3+alpha*pen
                np.testing.assert_allclose(energy,target,atol=1e-10,rtol=0)
                h,j=qubo_to_ising_coefficients(q,sizes,alpha);z=1-2*bits
                iz=sum(v*z[:,i] for i,v in h.items())+sum(v*z[:,i]*z[:,k] for (i,k),v in j.items())
                np.testing.assert_allclose(iz-iz[0],target-target[0],atol=1e-10,rtol=0)
                std=OpenQARPStandardQAOA(q,sizes,alpha)
                ref=independent_state(mat,sizes,.31,.27,False)
                np.testing.assert_allclose(std.probabilities([.31],[.27]),abs(ref)**2,atol=1e-10,rtol=0)

    def test_depth2_and_depth3_full_and_compact(self):
        for sizes in [(3,3),(3,3,3)]:
            n=sum(sizes);q=np.triu(np.random.default_rng(n).normal(0,.2,(n,n)))
            xy=OpenQARPXYQAOA(q,sizes);compact=OpenQARPCompactXYQAOA(q,sizes)
            for depth in [2,3]:
                gs=[.31,.7,1.1][:depth];bs=[.27,.4,.9][:depth]
                ref=independent_state(q,sizes,gs,bs,True)
                sv=xy.statevector(gs,bs)
                np.testing.assert_allclose(abs(sv)**2,abs(ref)**2,atol=1e-10,rtol=0)
                self.assertLess(abs(1-np.sum(abs(sv[xy.feasible_indices])**2)),1e-10)
                np.testing.assert_allclose(compact.feasible_probabilities(gs,bs),abs(ref[xy.feasible_indices])**2,atol=1e-10,rtol=0)
                # Explicit penalty cost in the independent X reference.
                mat=q.copy();start=0
                for size in sizes:
                    for a in range(start,start+size):
                        mat[a,a]-=5.
                        for b in range(a+1,start+size):mat[a,b]+=10.
                    start+=size
                ref=independent_state(mat,sizes,gs,bs,False)
                np.testing.assert_allclose(OpenQARPStandardQAOA(q,sizes,5.).probabilities(gs,bs),abs(ref)**2,atol=1e-10,rtol=0)

    def test_registered_data_fm_and_adaptive_rule(self):
        cfg=load_config(['-c',str(CODE/'config_multilayer_filter.json')]);p=load_problem(cfg['problem'])
        self.assertEqual((p.n_bits,len(p.categories)),(24,4096))
        self.assertEqual(len(set(p.initial_dataset(42,40))),40)
        np.testing.assert_array_equal(p.patterns.reshape(-1,6,4).sum(2),1)
        ids=p.initial_dataset(42,40);ys=[p.evaluate_id(i) for i in ids]
        m=train_fm(p.patterns[ids].astype(np.float32),ys,24,cfg['fm'],42)
        m2=train_fm(p.patterns[ids].astype(np.float32),ys,24,cfg['fm'],42)
        self.assertEqual(model_sha256(m),model_sha256(m2))
        _,q,b=fm_to_qubo(m)
        with torch.no_grad():pred=m(torch.tensor(p.patterns,dtype=torch.float32)).numpy()
        np.testing.assert_allclose(pred,np.einsum('bi,ij,bj->b',p.patterns,q,p.patterns)+b,atol=1e-4,rtol=0)
        t=AdaptivePenaltyTracker();self.assertEqual(t.update(.5),2.);self.assertEqual(t.update(.8),2.5);self.assertEqual(t.update(1.),2.125)
        s=qubo_to_ising_scale(q);h,j=qubo_to_ising_coefficients(q)
        self.assertAlmostEqual(s,max(abs(v) for v in list(h.values())+list(j.values())))

    def test_checkpoint_resume_preserves_results(self):
        cfg=load_config(['-c',str(CODE/'config_multilayer_filter.json')]);cfg=copy.deepcopy(cfg)
        cfg['problem']['bbo_cycles']=2
        cfg['methods']=['Exact-FM','Adaptive-FMQA','LargePenalty-FMQA','XY-FMQAOA']
        cfg['sa'].update(num_reads=10,num_sweeps=10)
        cfg['qaoa']['angle_search_mode']='legacy9'
        p=load_problem(cfg['problem'])
        with tempfile.TemporaryDirectory() as tmp:
            cp=Path(tmp)/'json/seed_42.json';cp.parent.mkdir()
            full=main.run_seed(42,p,cfg)
            dispatch=main.run_single_step_sampler
            def interrupt(**kwargs):
                if kwargs['method']=='Adaptive-FMQA' and kwargs['step']==1:raise RuntimeError('test interrupt')
                return dispatch(**kwargs)
            with patch.object(main,'run_single_step_sampler',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'test interrupt'):main.run_seed(42,p,cfg,checkpoint_path=cp)
            partial=json.loads(cp.read_text())
            self.assertEqual(partial['methods']['Adaptive-FMQA']['status'],'running')
            self.assertEqual(len(partial['methods']['Adaptive-FMQA']['history']),1)
            resumed=main.run_seed(42,p,cfg,existing_rec=partial,checkpoint_path=cp)
            for method in cfg['methods']:
                for key in ['final_best_objective','total_bb_evaluations','step0_fm_sha256']:
                    self.assertEqual(full['methods'][method][key],resumed['methods'][method][key])
                self.assertEqual([h['chosen_candidate_id'] for h in full['methods'][method]['history']],
                                 [h['chosen_candidate_id'] for h in resumed['methods'][method]['history']])




class EndToEndAudit(unittest.TestCase):
    def test_all_five_methods_and_postprocessing(self):
        import csv
        from evaluator import CategoricalTableProblem
        from analyze_bbo import audit, render
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            schema={'parameters':[{'options':['a','b','c']},{'options':['a','b','c']}]}
            (path/'schema.json').write_text(json.dumps(schema))
            with (path/'data.csv').open('w',newline='') as f:
                csv.writer(f).writerows([list(c)+[float(i)] for i,c in enumerate(itertools.product(['a','b','c'],repeat=2))])
            p=CategoricalTableProblem(path/'data.csv',path/'schema.json')
            cfg=load_config(['-c',str(CODE/'config_multilayer_filter.json')]);cfg=copy.deepcopy(cfg)
            cfg['methods']=['Exact-FM','Adaptive-FMQA','LargePenalty-FMQA','Penalty-FMQAOA','XY-FMQAOA']
            cfg['seeds']=[42,101,2024,7,19];cfg['problem'].update(initial_samples=4,bbo_cycles=2)
            cfg['qaoa'].update(angle_search_mode='powell49',standard_angle_search_mode='powell49')
            cfg['methods'] += [f'{fam}(p={d},mean,powell49)' for d in [2,3] for fam in ['Penalty-FMQAOA','XY-FMQAOA']]
            cfg['sa'].update(num_reads=10,num_sweeps=10)
            records=[main.run_seed(seed,p,cfg) for seed in cfg['seeds']]
            audit(records,cfg,p)
            corrupted=copy.deepcopy(records);corrupted[0]['methods']['XY-FMQAOA']['history'][0]['total_bb_evaluations']+=1
            with self.assertRaises(AssertionError):audit(corrupted,cfg,p)
            render(records,cfg,path)
            for fmt in ['pdf','svg','png']:
                self.assertEqual(len(list((path/fmt).glob(f'*.{fmt}'))),5)
                self.assertTrue(all(f.stat().st_size>1000 for f in (path/fmt).iterdir()))

if __name__=='__main__':unittest.main()
