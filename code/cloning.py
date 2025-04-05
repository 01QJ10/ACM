import jax
import jax.numpy as jnp
import numpy as np
from loss import fidelity  # Ensure that loss.fidelity is defined elsewhere

def orthogonal_state(psi: jnp.ndarray) -> jnp.ndarray:
    """
    Given a 2-dimensional normalized state psi, returns an orthogonal normalized state psi_perp.
    
    Parameters:
        psi (jnp.ndarray): A normalized state vector (at least 2-dimensional).
        
    Returns:
        jnp.ndarray: The normalized state orthogonal to psi.
    """
    psi = psi[:2]  # Ensure we only use the first two components
    a, b = psi[0], psi[1]
    psi_perp = jnp.array([-jnp.conjugate(b), jnp.conjugate(a)], dtype=jnp.complex64)
    norm = jnp.linalg.norm(psi_perp)
    effective_psi = jax.lax.cond(norm > 0, lambda _: psi_perp / norm, lambda _: psi_perp, operand=None)
    return effective_psi

def buzek_hillery_clone(effective_psi: jnp.ndarray):
    """
    Applies the Buzek-Hillery cloning transformation on an effective qubit state.
    
    The transformation is:
        |ψ⟩ |R⟩ |M⟩ →
          √(2/3) |ψ⟩ |ψ⟩ |ψ⊥⟩  −  √(1/6)[|ψ⟩ |ψ⊥⟩ + |ψ⊥⟩ |ψ⟩] |ψ⟩.
    
    Parameters:
        effective_psi (jnp.ndarray): A normalized 2-dimensional state vector.
        
    Returns:
        tuple: (rho_AB, rho_A, rho_B) where:
            - rho_AB is the 4×4 density matrix for the joint two-qubit (clone) state,
            - rho_A and rho_B are the 2×2 reduced density matrices for clones A and B respectively.
    """
    # Normalize the input state
    psi = effective_psi / jnp.linalg.norm(effective_psi)
    psi_perp = orthogonal_state(psi)

    # Cloning coefficients
    coeff1 = jnp.sqrt(2 / 3)
    coeff2 = jnp.sqrt(1 / 6)
    
    # Basis ordering: Clone A ⊗ Clone B ⊗ Machine, each in C².
    term1 = coeff1 * jnp.kron(jnp.kron(psi, psi), psi_perp)
    term2 = -coeff2 * (jnp.kron(jnp.kron(psi, psi_perp), psi) +
                       jnp.kron(jnp.kron(psi_perp, psi), psi))
    
    psi_out = term1 + term2
    rho_total = jnp.outer(psi_out, jnp.conjugate(psi_out))

    # Partial trace over the machine subsystem (3 subsystems of dimension 2)
    rho_tensor = jnp.reshape(rho_total, (2, 2, 2, 2, 2, 2))
    # Trace out the machine indices (axes 2 and 5)
    rho_AB = jnp.trace(rho_tensor, axis1=2, axis2=5)
    
    # Obtain reduced density matrices for clones A and B
    rho_A = jnp.trace(rho_AB, axis1=1, axis2=3)
    rho_B = jnp.trace(rho_AB, axis1=0, axis2=2)
    
    # Optionally, rho_AB can be reshaped to 4x4 if needed:
    rho_AB = rho_AB.reshape(4, 4)
    
    return rho_AB, rho_A, rho_B