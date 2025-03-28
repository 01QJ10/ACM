import numpy as np
import pandas as pd
import torch
from generate_qutrit_state import save_states
import os

# Set the seed for reproducibility
seed = 4
np.random.seed(seed)
torch.manual_seed(seed)

def generate_random_almost_qubit_qutrit(overlap_val=0.9):
    """
    Generates a random normalized 3-dimensional complex vector (qutrit)
    that has an exact overlap (overlap_val) with the qubit subspace
    (i.e. sqrt(|z1|^2 + |z2|^2) = overlap_val).
    """
    # Check that the overlap_val is valid (must be in [0, 1]).
    if not 0 <= overlap_val <= 1:
        raise ValueError("overlap_val must be between 0 and 1")
    
    # Set amplitude for the first component:
    a_amp = np.sqrt(1 - overlap_val**2)
    # Choose a random phase for a
    a_phase = np.exp(1j * np.random.uniform(0, 2*np.pi))
    a = a_amp * a_phase

    # For the qubit subspace (components b and c), generate a random 2D complex vector:
    sub = np.random.randn(2) + 1j * np.random.randn(2)
    sub /= np.linalg.norm(sub)  # normalize to unit length
    sub *= overlap_val        # scale to have norm equal to overlap_val

    # Construct the full state vector:
    state = np.array([a, sub[0], sub[1]])
    return state
        
num_states = 2000
states = []
p_vals = np.arange(0.01, 0.51, 0.01)
for p in p_vals:
    print(p)
    states = [generate_random_almost_qubit_qutrit(1 - p) for _ in range(num_states)]
    os.makedirs(f"../noise/check/{seed}", exist_ok=True)
    if p < 0.1:
        save_states(f"../noise/check/{seed}/p_0_0{int(p * 100)}.txt", np.array(states))
    else:
        save_states(f"../noise/check/{seed}/p_0_{int(p * 100)}.txt", np.array(states))