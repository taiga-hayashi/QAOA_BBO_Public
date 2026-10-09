import unittest
import numpy as np
from run_angles import restrict,batch_best,OpenQARPXYQAOA,all_bitstrings_lsb,one_hot_penalty

class Checks(unittest.TestCase):
    def test_restriction_all_binary_states(self):
        rng=np.random.default_rng(42);q=np.triu(rng.normal(size=(23,23)))
        free=[0,1,2,19,20,21];fixed=np.zeros(23);fixed[17]=1;sub,off=restrict(q,.4,free,fixed)
        bits=all_bitstrings_lsb(6);full=np.tile(fixed,(64,1));full[:,free]=bits
        np.testing.assert_allclose(np.einsum('bi,ij,bj->b',bits,sub,bits)+off,np.einsum('bi,ij,bj->b',full,q,full)+.4,atol=1e-12)
    def test_XY_preserves_N6_N9_at_test_angles(self):
        for n in [6,9]:
            q=np.diag(np.arange(n)/10);q[0,4]=.7;backend=OpenQARPXYQAOA(q,[3]*(n//3));bits=all_bitstrings_lsb(n)
            valid=one_hot_penalty(bits,[3]*(n//3))==0
            for gamma,beta in [(0,0),(.31,.27),(-.43,.61),(2*np.pi,np.pi)]:
                p=np.abs(backend.statevector([gamma],[beta]))**2
                self.assertLess(abs(p.sum()-1),1e-10);self.assertLess(abs(p[valid].sum()-1),1e-10)
    def test_exact_best_sample_formula(self):
        gap,success=batch_best(np.array([.2,.3]),np.array([1.,3.]),2)
        self.assertAlmostEqual(success,.75)
        self.assertAlmostEqual(gap,(.39*2)/.75)

if __name__=='__main__':unittest.main()
