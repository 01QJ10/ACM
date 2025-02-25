import numpy as np

def load_qutrit_states(filename):
    """
    Loads qutrit states from a text file.
    Expects each row to have 6 columns: [Re(z0), Re(z1), Re(z2), Im(z0), Im(z1), Im(z2)].
    Returns an array of shape (num_states, 3) of complex numbers.
    """
    data = np.loadtxt(filename)
    # Split the 6 columns into two groups of 3 (real and imaginary parts) and reassemble.
    real_part = data[:, :3]
    imag_part = data[:, 3:]
    complex_states = real_part + 1j * imag_part
    return complex_states



if __name__ == '__main__':
    train_states = load_qutrit_states("data/train.txt")
    test_states = load_qutrit_states("data/test.txt")
    print(type(test_states))
    print(f"Loaded {len(train_states)} training and {len(test_states)} test qutrit states.")