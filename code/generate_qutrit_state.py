# generate_qutrit_states.py
import numpy as np
from sklearn.model_selection import train_test_split

def generate_random_qutrit():
    """Generates a random normalized 3-dimensional complex vector (qutrit)."""
    state = np.random.randn(3) + 1j * np.random.randn(3)
    return state / np.linalg.norm(state)

num_states = 5000
states = np.array([generate_random_qutrit() for _ in range(num_states)])

# Split into 80% train and 20% test
train_states, test_states = train_test_split(states, test_size=0.2, random_state=42)

def generate_random_pseudo_qutrit():
    """Generates a random normalized 3-dimensional complex vector (qutrit)
    that lies completely on the qubit subspace (i.e. its first element is 0).
    """
    # Generate a random normalized 2-dimensional complex vector for the qubit subspace.
    substate = np.random.randn(2) + 1j * np.random.randn(2)
    substate = substate / np.linalg.norm(substate)
    
    # Create a 3-dimensional vector with the first element set to 0.
    state = np.zeros(3, dtype=complex)
    state[1:] = substate
    return state

num_states = 5000
states = np.array([generate_random_pseudo_qutrit() for _ in range(num_states)])

# Split into 80% train and 20% test
train_states, test_states = train_test_split(states, test_size=0.2, random_state=42)

def generate_random_almost_qubit_qutrit(overlap_val=0.9):
    """Generates a random normalized 3-dimensional complex vector (qutrit)
    that is almost a qubit, meaning its overlap with the qubit subspace 
    (i.e. sqrt(|z1|^2 + |z2|^2)) is at least p.
    """
    while True:
        state = np.random.randn(3) + 1j * np.random.randn(3)
        state /= np.linalg.norm(state)
        # Compute the overlap with the qubit subspace (elements 1 and 2).
        overlap = np.sqrt(np.abs(state[1])**2 + np.abs(state[2])**2)
        if overlap >= overlap_val:
            return state
num_states = 2500
states = np.array([generate_random_almost_qubit_qutrit() for _ in range(num_states)])

# Split into 80% train and 20% test
train_states, test_states = train_test_split(states, test_size=0.2, random_state=42)

def save_states(filename, states):
    """
    Saves an array of qutrit states to a text file.
    Each state is stored as 7 floats:
      [Re(z0), Re(z1), Re(z2), Im(z0), Im(z1), Im(z2), overlap]
    where overlap = sqrt(|z1|^2 + |z2|^2).
    The states are sorted in descending order by the overlap.
    """
    # Compute real and imaginary parts.
    re = np.real(states)
    im = np.imag(states)
    # Compute overlap for each state: sqrt(|b|^2 + |c|^2) where b and c are the 2nd and 3rd element.
    overlap = np.sqrt(np.abs(states[:, 1])**2 + np.abs(states[:, 2])**2).reshape(-1, 1)
    
    # Concatenate into a data array with 7 columns.
    data = np.hstack([re, im, overlap])
    
    # Sort rows in descending order by the overlap (last column).
    sorted_indices = np.argsort(data[:, -1])[::-1]
    sorted_data = data[sorted_indices]
    
    np.savetxt(filename, sorted_data, fmt='%.6f')

if __name__ == '__main__':
    save_states("../data/pseudo_train_80.txt", train_states)
    save_states("../data/pseudo_test_80.txt", test_states)
    print("Train and test qutrit states saved.")