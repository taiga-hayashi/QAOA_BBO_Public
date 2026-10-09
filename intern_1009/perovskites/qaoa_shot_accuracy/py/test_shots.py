import unittest
import numpy as np
from run_shots import draw,build_penalty_bqm,matrix,_build_qubo_energies

class Checks(unittest.TestCase):
    def test_cdf_boundaries_and_nested_prefix(self):
        cdf=np.array([.2,.2,.9,1.]);u=np.array([0,.199,.2,.899,.9,.999])
        np.testing.assert_array_equal(draw(cdf,u),[0,0,2,2,3,3])
        np.testing.assert_array_equal(draw(cdf,u[:3]),draw(cdf,u)[:3])
    def test_penalized_energies_all_states(self):
        q=np.diag([.1,-.2,.3,.5]);q[0,2]=.7
        bqm=build_penalty_bqm({(i,j):q[i,j] for i in range(4) for j in range(i,4)},[2,2],5,.6)
        qd,bias=bqm.to_qubo();e=_build_qubo_energies(matrix(qd,4),bias,np.dtype('float64'))
        bits=((np.arange(16)[:,None]>>np.arange(4))&1)
        expected=np.einsum('bi,ij,bj->b',bits,q,bits)+.6+5*((bits[:,:2].sum(1)-1)**2+(bits[:,2:].sum(1)-1)**2)
        np.testing.assert_allclose(e,expected,atol=1e-12)

if __name__=='__main__':unittest.main()
