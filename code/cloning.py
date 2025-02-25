import jax
import jax.numpy as jnp
import numpy as np
from code.loss import fidelity
def orthogonal_state(psi):
    """
    Given a 2-dimensional normalized state psi, returns an orthogonal normalized state psi_perp.
    """
    psi = psi[:2]
    a, b = psi[0], psi[1]
    psi_perp = jnp.array([-jnp.conjugate(b), jnp.conjugate(a)], dtype=jnp.complex64)
    norm = jnp.linalg.norm(psi_perp)
    effective_psi = jax.lax.cond(norm > 0, lambda _: psi_perp / norm, lambda _: psi_perp, operand=None)
    return effective_psi


def buzek_hillery_clone(effective_psi):
    """
    Applies the Buzek-Hillery cloning transformation on the effective qubit state.
    
    The transformation is:
        |ψ⟩ |R⟩ |M⟩ →
          √(2/3) |ψ⟩ |ψ⟩ |ψ⊥⟩  −  √(1/6)[|ψ⟩ |ψ⊥⟩ + |ψ⊥⟩ |ψ⟩] |ψ⟩.
    
    Parameters:
        effective_psi (jnp.ndarray): A normalized 2-dimensional state vector.
    
    Returns:
        tuple: (rho_A, rho_B) where each is a 2×2 density matrix for clones A and B.
    """
    # Ensure normalization
    psi = effective_psi / jnp.linalg.norm(effective_psi)
    psi_perp = orthogonal_state(psi)
    # print(psi)
    # print(psi_perp)
    coeff1 = jnp.sqrt(2/3)
    coeff2 = jnp.sqrt(1/6)
    
    # Basis ordering: Clone A ⊗ Clone B ⊗ Machine, each in C².
    term1 = coeff1 * jnp.kron(jnp.kron(psi, psi), psi_perp)
    term2 = - coeff2 * (jnp.kron(jnp.kron(psi, psi_perp), psi) +
                        jnp.kron(jnp.kron(psi_perp, psi), psi))
    
    psi_out = term1 + term2
    # print("psi_out:", psi_out)
    rho_total = jnp.outer(psi_out, jnp.conjugate(psi_out))

    # Partial trace over the machine subsystem.
    # Total Hilbert space dims: 2 x 2 x 2. Reshape rho_total to a 6-index tensor.
    rho_tensor = jnp.reshape(rho_total, (2, 2, 2, 2, 2, 2))
    # Trace out the machine: sum over the third and sixth indices (they represent the machine).
    # Using index labels: (i, a, k; j, b, n) with k and n being the machine indices.
    # We set n = k and sum over k.
    # rho_AB = jnp.einsum('ijkabc->ijab', rho_tensor)
    rho_AB = jnp.trace(rho_tensor, axis1=2, axis2=5)
    # Obtain reduced density matrix for clone A:
    # ρ_A[i, j] = sum_{a} ρ_AB[i, a, j, a]
    # rho_A = jnp.einsum('iaja->ij', rho_AB)
    # # Similarly, for clone B: ρ_B[a, b] = sum_{i} ρ_AB[i, a, i, b]
    # rho_B = jnp.einsum('iaib->ab', rho_AB)
    rho_A = jnp.trace(rho_AB, axis1=1, axis2=3)
    rho_B = jnp.trace(rho_AB, axis1=0, axis2=2)
    # print("rho_total:", rho_total)
    rho_AB = rho_AB.reshape(4, 4)
    # print("rho_AB:", rho_AB)
    # print("Norm of rho_A:", jnp.linalg.norm(rho_A))
    # print("Norm of rho_B:", jnp.linalg.norm(rho_B))
    # print(fidelity(psi, rho_B))
    
    return rho_AB, rho_A, rho_B