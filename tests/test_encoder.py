import sys
import unittest
from pathlib import Path

import jax.numpy as jnp

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from encoder import encode_qutrit


class TestEncoder(unittest.TestCase):
    def test_encoding_returns_state_and_unitary(self):
        qutrit_state = jnp.array([1 + 0j, 0 + 0j, 0 + 0j])
        weights = {str(i): 0.0 for i in range(1, 9)}

        encoded_state, encoder_unitary = encode_qutrit(qutrit_state, weights)

        self.assertEqual(encoded_state.shape, (3,))
        self.assertEqual(encoder_unitary.shape, (3, 3))
        self.assertTrue(jnp.allclose(encoded_state, qutrit_state))
        self.assertTrue(jnp.allclose(encoder_unitary, jnp.eye(3)))


if __name__ == "__main__":
    unittest.main()
