"""Regression checks for the shared OpenQARP Ring implementation."""
import sys
import unittest
from pathlib import Path
import numpy as np
import qarpx as qx
from scipy.linalg import expm

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
from qarp_backend import OpenQARPXYQAOA


class RingTests(unittest.TestCase):
    def test_heterogeneous_ring_edges(self):
        backend = OpenQARPXYQAOA(np.zeros((23, 23)), [16, 3, 4])
        expected = [(i, i+1) for i in range(15)] + [(15, 0), (16, 17), (17, 18), (18, 16), (19, 20), (20, 21), (21, 22), (22, 19)]
        self.assertEqual(backend.mixer_edges, expected)
        self.assertEqual(backend.onehot_penalty_lambda, 0)
        self.assertEqual(len(backend.feasible_indices), 192)

    def test_small_groups_do_not_double_count_edges(self):
        backend = OpenQARPXYQAOA(np.zeros((6, 6)), [1, 2, 3])
        self.assertEqual(backend.mixer_edges, [(1, 2), (3, 4), (4, 5), (5, 3)])

    def test_pair_gate_matches_exact_xx_yy_unitary(self):
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        beta = 0.37
        reference = expm(-0.5j * beta * (np.kron(x, x) + np.kron(y, y)))
        circuit = qx.SimpleBlock(2, 'pair rotation check')
        circuit.rxx(0, 1, beta);circuit.ryy(0, 1, beta);circuit.build()
        for index in range(4):
            initial = np.eye(4, dtype=complex)[:, index].copy()
            state = qx.QarpSimulator().statevector(circuit.commands(), 2, initial_state=initial)
            np.testing.assert_allclose(state, reference[:, index], atol=1e-12, rtol=0)

    def test_invalid_inputs(self):
        for q, blocks in [(np.zeros((4, 4)), [3]), (np.zeros((2, 3)), [3]), (np.zeros((3, 3)), [0, 3]), (np.zeros((3, 3)), [1.5, 1.5])]:
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):OpenQARPXYQAOA(q, blocks)
        q = np.zeros((3, 3));q[2, 0] = 1
        with self.assertRaises(ValueError):OpenQARPXYQAOA(q, [3])
        backend = OpenQARPXYQAOA(np.zeros((3, 3)), [3])
        for gamma, beta in [([], []), ([1], [1, 2]), ([float('nan')], [0])]:
            with self.assertRaises(ValueError):backend.build_circuit(gamma, beta)
        with self.assertRaises(ValueError):OpenQARPXYQAOA(np.zeros((3, 3)), [3], mixer_topology='invalid')

    def test_complete_mode_reproduces_old_gate_order(self):
        q = np.diag(np.arange(6)/7);q[0,4]=0.19
        backend = OpenQARPXYQAOA(q, [4, 2], mixer_topology='complete')
        gamma, beta = 0.23, 0.41
        old = qx.SimpleBlock(6, 'legacy complete reference')
        for qubit, coefficient in backend.linear.items():old.rz(qubit, 2*coefficient*gamma)
        for (left,right), coefficient in backend.quadratic.items():old.rzz(left,right,2*coefficient*gamma)
        for block in [[0, 1, 2, 3], [4, 5]]:
            for k, left in enumerate(block):
                for right in block[k+1:]:old.rxx(left, right, beta);old.ryy(left, right, beta)
        old.build()
        expected = qx.QarpSimulator().statevector(old.commands(), 6, initial_state=backend.initial_state)
        np.testing.assert_allclose(backend.statevector([gamma], [beta]), expected, atol=1e-12, rtol=0)

    def test_solver_records_new_topology(self):
        from qaoa_solver import solve_xy_qaoa
        result=solve_xy_qaoa({(0,0):0.1,(1,1):-0.2,(0,2):0.3},[2,2],maxiter=2)
        self.assertEqual(result['xy_mixer_topology'],'ring')
        self.assertEqual(result['xy_onehot_penalty_lambda'],0)
        self.assertLess(abs(result['feasibility_rate']-1),1e-10)

    def test_singleton_and_two_choice_state_preservation(self):
        backend = OpenQARPXYQAOA(np.diag([0.1, 0.2, 0.3]), [1, 2])
        state = backend.statevector([0.31], [0.27])
        self.assertLess(abs(1-np.sum(np.abs(state[backend.feasible_indices])**2)), 1e-10)


if __name__ == '__main__':unittest.main()
