import sys
import unittest
from pathlib import Path

import jax.numpy as jnp

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from loss import fidelity, loss_function


class TestLoss(unittest.TestCase):
    def test_fidelity_pure_state_against_density_matrix(self):
        state = jnp.array([1 + 0j, 0 + 0j])
        rho = jnp.outer(state, jnp.conjugate(state))
        self.assertAlmostEqual(float(fidelity(state, rho)), 1.0, places=5)

    def test_loss_symmetry_for_perfect_density_clones(self):
        original = jnp.array([1 + 0j, 0 + 0j, 0 + 0j])
        rho = jnp.outer(original, jnp.conjugate(original))
        loss = loss_function(original, rho, rho)
        self.assertAlmostEqual(float(loss), -1.0, places=5)


if __name__ == "__main__":
    unittest.main()
