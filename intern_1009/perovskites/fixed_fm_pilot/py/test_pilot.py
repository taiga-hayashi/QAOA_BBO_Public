import unittest
import numpy as np
import torch
from run_pilot import OpenQARPStandardQAOA, matrix, build_penalty_bqm, _build_qubo_energies, _sample_statevector_streaming, score_candidates
from qarp_backend import all_bitstrings_lsb, one_hot_penalty, qubo_to_ising_coefficients
from fm import TorchFM
from fm_to_qubo import fm_to_qubo


class PilotChecks(unittest.TestCase):
    def test_standard_statevector_against_independent_gate_math(self):
        q=np.diag(np.arange(6)/10);q[0,4]=0.2
        sizes=[3,3];gamma,beta=0.31,0.27;lam=1.7
        backend=OpenQARPStandardQAOA(q,sizes,lam)
        bits=all_bitstrings_lsb(6)
        energy=np.einsum('bi,ij,bj->b',bits,q,bits)+lam*one_hot_penalty(bits,sizes)
        ref=np.exp(-1j*gamma*energy)/8
        for qubit in range(6):
            for i in range(64):
                if i&(1<<qubit):continue
                j=i|(1<<qubit);a,b=ref[i],ref[j]
                ref[i]=np.cos(beta)*a-1j*np.sin(beta)*b
                ref[j]=np.cos(beta)*b-1j*np.sin(beta)*a
        bqm=build_penalty_bqm({(i,j):q[i,j] for i in range(6) for j in range(i,6)},sizes,lam,0)
        qd,offset=bqm.to_qubo();total=matrix(qd,6)
        global_offset=offset+np.trace(total)/2+np.triu(total,1).sum()/4
        ref*=np.exp(1j*gamma*global_offset)
        np.testing.assert_allclose(backend.statevector([gamma],[beta]),ref,atol=1e-12,rtol=0)

    def test_streamed_energy_matches_dense_and_penalty(self):
        q=np.diag([0.1,-0.2,0.3,0.7]);q[0,2]=0.8
        bqm=build_penalty_bqm({(i,j):q[i,j] for i in range(4) for j in range(i,4)},[2,2],3.2,0.9)
        qd,offset=bqm.to_qubo();total=matrix(qd,4);bits=all_bitstrings_lsb(4)
        expected=np.einsum('bi,ij,bj->b',bits,q,bits)+0.9+3.2*one_hot_penalty(bits,[2,2])
        np.testing.assert_allclose(_build_qubo_energies(total,offset,np.dtype('float64')),expected,atol=1e-12)

    def test_fm_qubo_binary_parity(self):
        torch.manual_seed(42);model=TorchFM(4,2).cpu();bits=all_bitstrings_lsb(4)
        qd,offset=fm_to_qubo(model);q=matrix(qd,4)
        with torch.no_grad():pred=model(torch.tensor(bits,dtype=torch.float32)).numpy()
        np.testing.assert_allclose(pred,np.einsum('bi,ij,bj->b',bits,q,bits)+offset,atol=2e-6)

    def test_sampler_and_duplicate_budget_policy(self):
        state=np.zeros(16,dtype=complex);state[[5,9]]=1/np.sqrt(2)
        indices=_sample_statevector_streaming(state,1000,42,4)
        self.assertEqual(set(indices),{5,9})
        np.testing.assert_array_equal(indices,_sample_statevector_streaming(state,1000,42,4))
        patterns=np.array([[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1]])
        samples=np.array([patterns[0],patterns[1],patterns[1],[0,0,0,0]])
        result=score_candidates(samples,{},np.diag([1,2,3,4]),0,[['a'],['b'],['c'],['d']],patterns,[0],np.array([1.,2.,3.,4.]),{'accepted_candidates':5})
        self.assertEqual(result['accepted_count'],1)
        self.assertEqual(result['unused_new_bb_budget'],4)
        self.assertEqual(result['raw_invalid_count'],1)
        self.assertEqual(result['initial_duplicate_count'],1)
        self.assertEqual(result['within_unseen_pool_duplicate_count'],1)


if __name__=='__main__':unittest.main()
