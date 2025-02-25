import unittest
import jax.numpy as jnp
from Code.loss import fidelity, loss_function

class TestLoss(unittest.TestCase):
    def test_fidelity_perfect(self):
        state = jnp.array([1+0j, 0+0j])
        self.assertAlmostEqual(fidelity(state, state), 1.0, places=5)
        
    def test_loss_symmetry(self):
        original = jnp.array([1+0j, 0+0j, 0+0j])
        clone_A = jnp.array([1+0j, 0+0j, 0+0j])
        clone_B = jnp.array([1+0j, 0+0j, 0+0j])
        loss = loss_function(original, clone_A, clone_B)
        # For perfect clones, the loss should be approximately 0.
        self.assertAlmostEqual(loss, 0.0, places=5)

if __name__ == '__main__':
    unittest.main()