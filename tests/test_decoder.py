import sys
import unittest
from pathlib import Path

import jax.numpy as jnp

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from cloning import buzek_hillery_clone
from decoder import (
    decode_qubit_to_qutrit,
    decode_two_qubits_to_qutrits,
    reduce_two_qutrit_state,
)
from embed import embed_single_qubit_density, embed_two_qubit_density


class TestDecoder(unittest.TestCase):
    def test_single_clone_decoder_shape_and_trace(self):
        rho_2x2 = jnp.array([[5 / 6, 0], [0, 1 / 6]], dtype=jnp.complex64)
        encoder_unitary = jnp.eye(3, dtype=jnp.complex64)

        decoded = decode_qubit_to_qutrit(embed_single_qubit_density(rho_2x2), encoder_unitary)

        self.assertEqual(decoded.shape, (3, 3))
        self.assertAlmostEqual(float(jnp.real(jnp.trace(decoded))), 1.0, places=5)

    def test_joint_decode_marginals_match_separate_decode(self):
        effective = jnp.array([1 / jnp.sqrt(2), 1j / jnp.sqrt(2)], dtype=jnp.complex64)
        rho_ab, rho_a, rho_b = buzek_hillery_clone(effective)
        encoder_unitary = jnp.array(
            [[0, 1, 0],
             [1, 0, 0],
             [0, 0, 1]],
            dtype=jnp.complex64,
        )

        separate_a = decode_qubit_to_qutrit(embed_single_qubit_density(rho_a), encoder_unitary)
        separate_b = decode_qubit_to_qutrit(embed_single_qubit_density(rho_b), encoder_unitary)
        joint = decode_two_qubits_to_qutrits(embed_two_qubit_density(rho_ab), encoder_unitary)
        joint_a = reduce_two_qutrit_state(joint, "A")
        joint_b = reduce_two_qutrit_state(joint, "B")

        self.assertEqual(joint.shape, (9, 9))
        self.assertAlmostEqual(float(jnp.real(jnp.trace(joint))), 1.0, places=5)
        self.assertTrue(jnp.allclose(joint_a, separate_a, atol=1e-5))
        self.assertTrue(jnp.allclose(joint_b, separate_b, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
