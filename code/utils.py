import numpy as np
import jax
import jax.numpy as jnp

def load_qutrit_states(filename: str) -> np.ndarray:
    """
    Load qutrit states from a text file.
    
    Each row in the file should contain 6 columns:
      [Re(z0), Re(z1), Re(z2), Im(z0), Im(z1), Im(z2)].
      
    Returns:
        A NumPy array of shape (num_states, 3) with complex numbers.
    """
    data = np.loadtxt(filename)
    real_part = data[:, :3]
    imag_part = data[:, 3:]
    complex_states = real_part + 1j * imag_part
    return complex_states

def hs_norm(U: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the normalized Hilbert–Schmidt inner product between two unitary operators.
    
    Args:
        U: A (d x d) unitary matrix.
        V: A (d x d) unitary matrix.
        
    Returns:
        The normalized inner product: |Tr(U† V)| / d.
    """
    d = U.shape[0]
    hs_inner = jnp.abs(jnp.trace(jnp.conjugate(U).T @ V))
    return hs_inner / d

def frobenius_norm(U: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Frobenius norm of the difference between two unitary operators.
    
    Args:
        U: A (d x d) unitary matrix.
        V: A (d x d) unitary matrix.
        
    Returns:
        The Frobenius norm ||U - V||_F.
    """
    diff = U - V
    return jnp.sqrt(jnp.sum(jnp.abs(diff)**2))

def main() -> None:
    # Load qutrit states from files.
    train_states = load_qutrit_states("data/train.txt")
    test_states = load_qutrit_states("data/test.txt")
    print(f"Loaded {len(train_states)} training and {len(test_states)} test qutrit states. Type: {type(test_states)}")
    
    # Define example unitary matrices.
    U = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    V = jnp.eye(2, dtype=jnp.complex64)
    
    # Compute norms.
    norm_hs_value = hs_norm(U, V)
    frob_norm_value = frobenius_norm(U, V)
    
    print("Normalized Hilbert–Schmidt inner product:", norm_hs_value)
    print("Frobenius norm of the difference:", frob_norm_value)

if __name__ == '__main__':
    main()