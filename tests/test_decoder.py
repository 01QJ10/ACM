import unittest
import jax.numpy as jnp
from Code.decoder import decode_qubit_to_qutrit

class TestDecoder(unittest.TestCase):
    def test_decoder_normalization(self):
        # Create a simple 4-dimensional state vector.
        state_4d = jnp.array([1+0j, 0+0j, 0+0j, 0+0j])
        # Use an identity matrix for the encoder unitary for testing purposes.
        encoder_unitary = jnp.eye(2, dtype=jnp.complex64)
        decoded = decode_qubit_to_qutrit(state_4d, encoder_unitary)
        self.assertAlmostEqual(jnp.linalg.norm(decoded), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()