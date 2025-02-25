import unittest
import jax.numpy as jnp
import numpy as np
import jax
import sys
sys.path.append('/Users/behlari/Desktop/FYP/QAM')
import os
from code.cloning import buzek_hillery_clone
from code.loss import fidelity

class TestCloning(unittest.TestCase):
    def test_cloning_normalization(self):
        qutrit_state = jnp.array([0+0j, 0-0j, 0-1j])
        # qutrit_state = jnp.array([0+0j, 0-0j, 0-1j])
        effective_part = qutrit_state[1:3]
        norm_eff = jnp.linalg.norm(effective_part)
        qubit_state = jax.lax.cond(norm_eff > 0, lambda _: effective_part / norm_eff, lambda _: effective_part, operand=None)
        clone_A, clone_B = buzek_hillery_clone(qubit_state)
        print(fidelity(qubit_state, clone_A))
        print(fidelity(qubit_state, clone_B))
        self.assertAlmostEqual(jnp.trace(clone_A), 1.0, places=5)
        self.assertAlmostEqual(jnp.trace(clone_B), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()