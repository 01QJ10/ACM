import numpy as np
from sklearn.model_selection import train_test_split
import argparse

def generate_random_qutrit() -> np.ndarray:
    """Generates a random normalized 3-dimensional complex vector (qutrit)."""
    state = np.random.randn(3) + 1j * np.random.randn(3)
    return state / np.linalg.norm(state)

def generate_random_pseudo_qutrit() -> np.ndarray:
    """Generates a random normalized qutrit that lies entirely in the qubit subspace (first element is 0)."""
    substate = np.random.randn(2) + 1j * np.random.randn(2)
    substate /= np.linalg.norm(substate)
    state = np.zeros(3, dtype=complex)
    state[1:] = substate
    return state

def generate_random_almost_qubit_qutrit(overlap_val: float = 0.9) -> np.ndarray:
    """
    Generates a random normalized qutrit that is almost a qubit.
    
    The state’s overlap with the qubit subspace (sqrt(|z1|^2 + |z2|^2)) is at least overlap_val.
    """
    while True:
        state = np.random.randn(3) + 1j * np.random.randn(3)
        state /= np.linalg.norm(state)
        overlap = np.sqrt(np.abs(state[1])**2 + np.abs(state[2])**2)
        if overlap >= overlap_val:
            return state

def save_states(filename: str, states: np.ndarray) -> None:
    """
    Saves an array of qutrit states to a text file.
    
    Each state is stored as 7 floats:
      [Re(z0), Re(z1), Re(z2), Im(z0), Im(z1), Im(z2), overlap],
    where overlap = sqrt(|z1|^2 + |z2|^2). The rows are sorted by decreasing overlap.
    """
    re = np.real(states)
    im = np.imag(states)
    overlap = np.sqrt(np.abs(states[:, 1])**2 + np.abs(states[:, 2])**2).reshape(-1, 1)
    data = np.hstack([re, im, overlap])
    sorted_indices = np.argsort(data[:, -1])[::-1]
    sorted_data = data[sorted_indices]
    np.savetxt(filename, sorted_data, fmt='%.6f')

def main(mode: str, num_states: int, test_size: float) -> None:
    """
    Generates qutrit states based on the specified mode and saves train/test splits.
    
    Args:
        mode: 'random', 'pseudo', or 'almost'
        num_states: Number of states to generate.
        test_size: Fraction of states to use as test set.
    """
    if mode == "random":
        states = np.array([generate_random_qutrit() for _ in range(num_states)])
    elif mode == "pseudo":
        states = np.array([generate_random_pseudo_qutrit() for _ in range(num_states)])
    elif mode == "almost":
        states = np.array([generate_random_almost_qubit_qutrit() for _ in range(num_states)])
    else:
        raise ValueError("Unknown mode. Choose from 'random', 'pseudo', or 'almost'.")
    
    train_states, test_states = train_test_split(states, test_size=test_size, random_state=42)
    save_states("../data/train_states.txt", train_states)
    save_states("../data/test_states.txt", test_states)
    print("Train and test qutrit states saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and save qutrit states.")
    parser.add_argument("--mode", type=str, default="random",
                        help="Type of state to generate: 'random', 'pseudo', or 'almost'")
    parser.add_argument("--num_states", type=int, default=5000,
                        help="Number of states to generate.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of states to use as test set.")
    args = parser.parse_args()
    main(args.mode, args.num_states, args.test_size)