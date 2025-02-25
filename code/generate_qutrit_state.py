# generate_qutrit_states.py
import numpy as np
from sklearn.model_selection import train_test_split

def generate_random_qutrit():
    """Generates a random normalized 3-dimensional complex vector (qutrit)."""
    state = np.random.randn(3) + 1j * np.random.randn(3)
    return state / np.linalg.norm(state)

num_states = 10000
states = np.array([generate_random_qutrit() for _ in range(num_states)])

# Split into 80% train and 20% test
train_states, test_states = train_test_split(states, test_size=0.2, random_state=42)

def save_states(filename, states):
    """
    Saves an array of qutrit states to a text file.
    Each state is stored as 6 floats: [Re(z0), Re(z1), Re(z2), Im(z0), Im(z1), Im(z2)]
    """
    # Prepare a 2D array where each row is a qutrit's real parts followed by its imaginary parts.
    data = np.hstack([np.real(states), np.imag(states)])
    np.savetxt(filename, data, fmt='%.6f')

if __name__ == '__main__':
    save_states("data/train.txt", train_states)
    save_states("data/test.txt", test_states)
    print("Train and test qutrit states saved.")