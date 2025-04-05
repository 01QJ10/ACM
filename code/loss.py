import jax
import jax.numpy as jnp

def fidelity(psi: jnp.ndarray, rho: jnp.ndarray) -> float:
    """
    Computes the fidelity between a pure state psi and a density matrix rho:
        F = ⟨psi| rho |psi⟩.
    
    Parameters:
        psi: A normalized state vector.
        rho: A density matrix.
    
    Returns:
        The fidelity (a real number).
    """
    return jnp.real(jnp.dot(jnp.conjugate(psi), jnp.dot(rho, psi)))

def loss_function(original_state: jnp.ndarray, clone_A: jnp.ndarray, clone_B: jnp.ndarray) -> float:
    """
    Calculates the cloning loss:
        L = 1 - F_A - F_B + (F_A - F_B)^2,
    where F_A and F_B are fidelities of the two clones with respect to the original state.
    
    Parameters:
        original_state: The original state vector.
        clone_A: Decoded clone A.
        clone_B: Decoded clone B.
    
    Returns:
        The loss value.
    """
    F_A = fidelity(original_state, clone_A)
    F_B = fidelity(original_state, clone_B)
    return 1 - F_A - F_B + (F_A - F_B) ** 2