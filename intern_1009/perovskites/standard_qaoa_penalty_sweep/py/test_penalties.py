import unittest
import numpy as np
from run_penalties import matrix,build_penalty_bqm,_build_qubo_energies,OpenQARPStandardQAOA

class Checks(unittest.TestCase):
    def test_high_penalty_cost_all_states(self):
        q=np.diag([.1,-.2,.3,.5,.6,.7]);q[0,4]=.9
        bits=((np.arange(64)[:,None]>>np.arange(6))&1)
        for alpha in [0,.01,5,100,1000]:
            bqm=build_penalty_bqm({(i,j):q[i,j] for i in range(6) for j in range(i,6)},[3,3],alpha,.4)
            qd,bias=bqm.to_qubo();energies=_build_qubo_energies(matrix(qd,6),bias,np.dtype('float64'))
            ref=np.einsum('bi,ij,bj->b',bits,q,bits)+.4+alpha*((bits[:,:3].sum(1)-1)**2+(bits[:,3:].sum(1)-1)**2)
            np.testing.assert_allclose(energies,ref,atol=1e-10)
    def test_standard_high_alpha_against_independent_phase_rx(self):
        q=np.diag(np.arange(6)/10);bits=((np.arange(64)[:,None]>>np.arange(6))&1);gamma,beta=.4,.8
        for alpha in [0,1000]:
            penalty=(bits[:,:3].sum(1)-1)**2+(bits[:,3:].sum(1)-1)**2
            ref=np.exp(-1j*gamma*(np.einsum('bi,ij,bj->b',bits,q,bits)+alpha*penalty))/8
            for qubit in range(6):
                for i in range(64):
                    if i&(1<<qubit):continue
                    j=i|(1<<qubit);a,b=ref[i],ref[j]
                    ref[i]=np.cos(beta)*a-1j*np.sin(beta)*b;ref[j]=np.cos(beta)*b-1j*np.sin(beta)*a
            state=OpenQARPStandardQAOA(q,[3,3],alpha).statevector([gamma],[beta])
            np.testing.assert_allclose(np.abs(state)**2,np.abs(ref)**2,atol=1e-11)

if __name__=='__main__':unittest.main()
