import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from reviewer_response_experiments import (
    fixed_projection_reconstruction_fidelity,
    fixed_subspace_clone_metrics,
)


class TestReviewerResponseExperiments(unittest.TestCase):
    def test_fixed_subspace_clone_on_exact_subspace_state(self):
        state = [0 + 0j, 1 + 0j, 0 + 0j]
        metrics = fixed_subspace_clone_metrics(state)

        self.assertAlmostEqual(metrics["fidelity_a"], 5 / 6, places=4)
        self.assertAlmostEqual(metrics["fidelity_b"], 5 / 6, places=4)

    def test_clone_free_reconstruction_upper_bound(self):
        state = [0 + 0j, 1 + 0j, 0 + 0j]
        fidelity = fixed_projection_reconstruction_fidelity(state)

        self.assertAlmostEqual(fidelity, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
