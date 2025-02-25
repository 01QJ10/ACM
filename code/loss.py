import jax
import jax.numpy as jnp

def fidelity(psi, rho):
    """
    Computes the fidelity between a pure state psi and a density matrix rho:
        F = ⟨psi| rho |psi⟩.
    
    Parameters:
        psi (jnp.ndarray): A normalized 2-dimensional state vector.
        rho (jnp.ndarray): A 2×2 density matrix.
    
    Returns:
        float: The fidelity.
    """
    return jnp.real(jnp.dot(jnp.conjugate(psi), jnp.dot(rho, psi)))

def loss_function(original_state, clone_A, clone_B):
    """
    Calculates the loss as:
        L = 1 - F_A - F_B + (F_A - F_B)^2,
    where F_A and F_B are the fidelities of the two clones with respect to the original state.
    
    Parameters:
        original_state (jnp.ndarray): The original qutrit state.
        clone_A (jnp.ndarray): Decoded clone A (qutrit state).
        clone_B (jnp.ndarray): Decoded clone B (qutrit state).
    
    Returns:
        float: Loss value.
    """
    F_A = fidelity(original_state, clone_A)
    F_B = fidelity(original_state, clone_B)
    loss = 1 - F_A - F_B + (F_A - F_B)**2
    return loss