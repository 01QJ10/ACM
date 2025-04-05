import unittest
import jax.numpy as jnp
import numpy as np
import jax
import sys
import os
import sys

sys.path.insert(1, '/Users/behlari/Desktop/FYP/QAM')
from code.cloning import buzek_hillery_clone
sys.path.insert(1, '/Users/behlari/Desktop/FYP/QAM')
from code.loss import fidelity


class TestCloning(unittest.TestCase):
    def test_cloning_normalization(self):
        # Create a qutrit state with no noise in the qubit subspace:
        # We assume the qubit subspace is given by indices 1 and 2.
        # Here we choose a state that is exactly in that subspace:
        qutrit_state = jnp.array([0+0j, 1+0j, 0+0j])
        
        # Extract the effective qubit state (indices 1 and 2)
        effective_part = qutrit_state[1:3]
        norm_eff = jnp.linalg.norm(effective_part)
        qubit_state = jax.lax.cond(norm_eff > 0, 
                                   lambda _: effective_part / norm_eff, 
                                   lambda _: effective_part, 
                                   operand=None)
        
        clone_AB, clone_A, clone_B = buzek_hillery_clone(qubit_state)

        # Print fidelities for visual inspection
        fid_A = fidelity(qubit_state, clone_A)
        fid_B = fidelity(qubit_state, clone_B)

        print("Fidelity clone A:", fid_A)
        print("Fidelity clone B:", fid_B)

        # Check that the reduced density matrices are normalized
        self.assertAlmostEqual(jnp.trace(clone_A), 1.0, places=5)
        self.assertAlmostEqual(jnp.trace(clone_B), 1.0, places=5)
        
        # When there is no noise, the fidelity should approach the optimal value for a 1->2 universal cloner.
        # For qubits, the known optimal fidelity is F = 5/6 ≈ 0.83333.
        self.assertAlmostEqual(fid_A, 5/6, places=4)
        self.assertAlmostEqual(fid_B, 5/6, places=4)

        

if __name__ == '__main__':
    unittest.main()
