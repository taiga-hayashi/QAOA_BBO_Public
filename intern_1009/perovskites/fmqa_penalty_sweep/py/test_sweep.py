import unittest
import numpy as np
from run_sweep import build_penalty_bqm, measure, PerovskitesEvaluator

class Checks(unittest.TestCase):
    def test_penalty_energy_all_small_states(self):
        for n in [6,9]:
            sizes=[3]*(n//3);q=np.diag(np.arange(n)*0.1);q[0,4]=0.7
            qd={(i,j):q[i,j] for i in range(n) for j in range(i,n)}
            for alpha in [0,0.01,100,1000]:
                lam=2.3*alpha;bqm=build_penalty_bqm(qd,sizes,lam,0.4)
                for idx in range(1<<n):
                    x=np.array([(idx>>i)&1 for i in range(n)])
                    violation=sum((x[g:g+3].sum()-1)**2 for g in range(0,n,3))
                    self.assertAlmostEqual(bqm.energy(dict(enumerate(x))),x@q@x+0.4+lam*violation,places=8)

    def test_metric_hand_counts_and_empty_case(self):
        bb=PerovskitesEvaluator();cats=[list(c) for c in bb.candidates()]
        patterns=np.array([bb.encode(c) for c in cats]);truth=np.arange(192,dtype=float)
        q=np.zeros((23,23));bits=np.array([patterns[0],patterns[1],patterns[1],patterns[2],np.zeros(23)])
        r=measure(bits,q,0,patterns,[0],truth,cats)
        self.assertEqual(r['empirical_raw_feasible_rate'],0.8)
        self.assertEqual(r['unique_unseen_candidates'],2)
        self.assertEqual(r['raw_unseen_true_top5pct_fraction'],0.6)
        self.assertEqual(r['batch_any_unseen_true_top5pct'],1)
        self.assertEqual(r['batch_at_least_five_unseen'],0)
        self.assertAlmostEqual(r['conditional_unseen_effective_diversity'],np.exp(-(2/3*np.log(2/3)+1/3*np.log(1/3))))
        empty=measure(np.zeros((5,23)),q,0,patterns,[0],truth,cats)
        self.assertIsNone(empty['conditional_unseen_mean_surrogate_gap'])
        self.assertEqual(empty['accepted_count'],0)

if __name__=='__main__':unittest.main()
