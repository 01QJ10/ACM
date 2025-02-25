import unittest
import jax.numpy as jnp
from code.encoder import encode_qutrit

class TestEncoder(unittest.TestCase):
    def test_encoding_normalization(self):
        qutrit_state = jnp.array([1+0j, 0+0j, 0+0j])
        weights = {'1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0,
                   '5': 0.0, '6': 0.0, '7': 0.0, '8': 0.0}
        qubit_state = encode_qutrit(qutrit_state, weights)
        # With zero weights, U becomes identity so the projection should remain unchanged.
        expected = qutrit_state[:2]
        self.assertTrue(jnp.allclose(qubit_state[:2], expected))

if __name__ == '__main__':
    unittest.main()