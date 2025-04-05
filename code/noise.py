import numpy as np
import os
import torch
from generate_qutrit_state import save_states

seed = 6
np.random.seed(seed)
torch.manual_seed(seed)

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

def main():
    num_states = 2000
    seed_dir = f"../noise/check/{seed}"
    os.makedirs(seed_dir, exist_ok=True)
    p_vals = np.arange(0.01, 1.01, 0.01)
    for p in p_vals:
        print(f"Generating for p = {p}")
        states = np.array([generate_random_almost_qubit_qutrit(1 - p) for _ in range(num_states)])
        if p < 0.1:
            filename = f"{seed_dir}/p_0_0{int(round(p, 2) * 100)}.txt"
        elif p == 0.29:
            filename = f"{seed_dir}/p_0_29.txt"
        elif p == 0.58:
            filename = f"{seed_dir}/p_0_58.txt"
        else:
            filename = f"{seed_dir}/p_0_{int(round(p, 2) * 100)}.txt"
        save_states(filename, states)

if __name__ == "__main__":
    main()