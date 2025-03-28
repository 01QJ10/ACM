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

import jax
import jax.numpy as jnp

def hs_norm(U, V):
    """
    Computes the normalized Hilbert–Schmidt inner product between two unitary operators.
    
    Args:
        U: A (d x d) unitary matrix.
        V: A (d x d) unitary matrix.
        
    Returns:
        A scalar representing the normalized inner product: |Tr(U† V)| / d.
    """
    d = U.shape[0]
    hs_inner = jnp.abs(jnp.trace(jnp.conjugate(U).T @ V))
    return hs_inner / d

def frobenius_norm(U, V):
    """
    Computes the Frobenius norm of the difference between two unitary operators.
    
    Args:
        U: A (d x d) unitary matrix.
        V: A (d x d) unitary matrix.
        
    Returns:
        A scalar representing the Frobenius norm ||U - V||_F.
    """
    diff = U - V
    # Frobenius norm: sqrt(sum(|diff|^2))
    return jnp.sqrt(jnp.sum(jnp.abs(diff)**2))

if __name__ == '__main__':
    train_states = load_qutrit_states("data/train.txt")
    test_states = load_qutrit_states("data/test.txt")
    print(type(test_states))
    print(f"Loaded {len(train_states)} training and {len(test_states)} test qutrit states.")
    U = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    V = jnp.eye(2, dtype=jnp.complex64)
    
    norm_hs = hs_norm(U, V)
    frob_norm = frobenius_norm(U, V)
    
    print("Normalized Hilbert–Schmidt inner product:", norm_hs)
    print("Frobenius norm of the difference:", frob_norm)