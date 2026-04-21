import jax.numpy as jnp

def decode_qubit_to_qutrit(cloned_qubit_state: jnp.ndarray, encoder_unitary: jnp.ndarray) -> jnp.ndarray:
    """
    Decodes an embedded one-clone qutrit density matrix with U_e^\dagger.
    
    Parameters:
        cloned_qubit_state (jnp.ndarray): A 3x3 embedded one-clone density matrix.
        encoder_unitary (jnp.ndarray): The 3×3 unitary matrix used in the encoding process.
        
    Returns:
        jnp.ndarray: A decoded 3x3 qutrit density matrix.
    """
    U_dag = jnp.conjugate(encoder_unitary.T)
    return U_dag @ cloned_qubit_state @ encoder_unitary


def decode_two_qubits_to_qutrits(rho_9x9: jnp.ndarray, encoder_unitary: jnp.ndarray) -> jnp.ndarray:
    """Decode an embedded two-clone 9x9 state with U_e^\dagger ⊗ U_e^\dagger."""
    U_dag = jnp.conjugate(encoder_unitary.T)
    joint_decoder = jnp.kron(U_dag, U_dag)
    joint_encoder = jnp.kron(encoder_unitary, encoder_unitary)
    return joint_decoder @ rho_9x9 @ joint_encoder


def reduce_two_qutrit_state(rho_9x9: jnp.ndarray, keep: str = "A") -> jnp.ndarray:
    """Trace one qutrit out of a two-qutrit 9x9 density matrix."""
    rho_tensor = jnp.reshape(rho_9x9, (3, 3, 3, 3))
    if keep == "A":
        return jnp.trace(rho_tensor, axis1=1, axis2=3)
    if keep == "B":
        return jnp.trace(rho_tensor, axis1=0, axis2=2)
    raise ValueError("keep must be 'A' or 'B'")
