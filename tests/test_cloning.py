import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from cloning import buzek_hillery_clone
from loss import fidelity


class TestCloning(unittest.TestCase):
    def test_cloning_normalization_and_qubit_fidelity(self):
        qutrit_state = jnp.array([0 + 0j, 1 + 0j, 0 + 0j])
        effective_part = qutrit_state[1:3]
        norm_eff = jnp.linalg.norm(effective_part)
        qubit_state = jax.lax.cond(
            norm_eff > 0,
            lambda _: effective_part / norm_eff,
            lambda _: effective_part,
            operand=None,
        )

        _, clone_a, clone_b = buzek_hillery_clone(qubit_state)
        fid_a = fidelity(qubit_state, clone_a)
        fid_b = fidelity(qubit_state, clone_b)

        self.assertAlmostEqual(float(jnp.real(jnp.trace(clone_a))), 1.0, places=5)
        self.assertAlmostEqual(float(jnp.real(jnp.trace(clone_b))), 1.0, places=5)
        self.assertAlmostEqual(float(fid_a), 5 / 6, places=4)
        self.assertAlmostEqual(float(fid_b), 5 / 6, places=4)


if __name__ == "__main__":
    unittest.main()
