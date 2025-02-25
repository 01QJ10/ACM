import jax
import jax.numpy as jnp

def decode_qubit_to_qutrit(cloned_qubit_state, encoder_unitary):
    """
    Decodes a cloned qubit state back into a qutrit state.
    
    The decoder is built as the tensor product of two conjugate transposes of the encoder's unitary:
        U_decoder = U† ⊗ U†
    The resulting 4-dimensional state is then projected onto a 3-dimensional subspace.
    
    Parameters:
        cloned_qubit_state (jnp.ndarray): Cloned state (as a 4-dimensional vector from a 2-qubit system).
        encoder_unitary (jnp.ndarray): The unitary used in encoding.
    
    Returns:
        jnp.ndarray: Decoded qutrit state (3-dimensional).
    """
    # Compute the conjugate transpose of the encoder unitary
    U_dag = jnp.conjugate(encoder_unitary.T)
    # Construct the decoder unitary via the tensor (Kronecker) product
    # decoder_unitary = jnp.kron(U_dag, U_dag)
    
    # Apply the decoder to the cloned state
    decoded_state_full = U_dag @ cloned_qubit_state @ encoder_unitary
    
    # Project to a 3-dimensional qutrit state (take first three components and normalize)
    # decoded_state = decoded_state_full[:3]
    # norm = jnp.linalg.norm(decoded_state)
    # if norm > 0:
    #     decoded_state = decoded_state / norm
    return decoded_state_full