import argparse
from pathlib import Path

import numpy as np
import torch
from generate_qutrit_state import save_states

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 6


def p_to_filename(p: float) -> str:
    return f"p_0_{int(round(p, 2) * 100):02d}.txt"


def generate_random_almost_qubit_qutrit(overlap_val: float = 0.9) -> np.ndarray:
    """
    Generates a random normalized qutrit that has an exact overlap with the qubit subspace.
    
    The overlap (sqrt(|z1|^2 + |z2|^2)) is set to overlap_val.
    """
    if not 0 <= overlap_val <= 1:
        raise ValueError("overlap_val must be between 0 and 1")
    a_amp = np.sqrt(1 - overlap_val**2)
    a_phase = np.exp(1j * np.random.uniform(0, 2 * np.pi))
    a = a_amp * a_phase
    sub = np.random.randn(2) + 1j * np.random.randn(2)
    sub /= np.linalg.norm(sub)
    sub *= overlap_val
    state = np.array([a, sub[0], sub[1]])
    return state

def main(seed: int = DEFAULT_SEED, num_states: int = 2000, output_root: Path | None = None,
         p_min: float = 0.01, p_max: float = 1.00, p_step: float = 0.01):
    np.random.seed(seed)
    torch.manual_seed(seed)
    output_root = output_root or (REPO_ROOT / "noise" / "check")
    seed_dir = output_root / str(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    p_vals = np.arange(p_min, p_max + 0.5 * p_step, p_step)
    for p in p_vals:
        print(f"Generating for p = {p}")
        states = np.array([generate_random_almost_qubit_qutrit(1 - p) for _ in range(num_states)])
        filename = seed_dir / p_to_filename(p)
        save_states(filename, states)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate exact-overlap noisy qutrit datasets.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-states", type=int, default=2000)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--p-min", type=float, default=0.01)
    parser.add_argument("--p-max", type=float, default=1.00)
    parser.add_argument("--p-step", type=float, default=0.01)
    args = parser.parse_args()
    main(args.seed, args.num_states, args.output_root, args.p_min, args.p_max, args.p_step)
