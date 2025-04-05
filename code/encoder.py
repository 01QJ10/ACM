import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

def get_generators() -> dict:
    """
    Returns 3×3 generators (analogous to the Pauli matrices for qubits)
    using a subset of the Gell-Mann matrices.
    
    The following generators are defined:
        lambda1 = [[0, 1, 0],
                   [1, 0, 0],
                   [0, 0, 0]]
        lambda2 = [[0, -i, 0],
                   [i,  0, 0],
                   [0,  0, 0]]
        lambda3 = [[1,  0, 0],
                   [0, -1, 0],
                   [0,  0, 0]]
        ... and additional generators lambda4 through lambda8.
    
    Returns:
        dict: A dictionary mapping string keys ('1' through '8') to the corresponding generator matrices.
    """
    lambda1 = jnp.array([[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, 0]], dtype=jnp.complex64)
    lambda2 = jnp.array([[0, -1j, 0],
                         [1j, 0, 0],
                         [0, 0, 0]], dtype=jnp.complex64)
    lambda3 = jnp.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, 0]], dtype=jnp.complex64)
    lambda4 = jnp.array([[0, 0, 1],
                         [0, 0, 0],
                         [1, 0, 0]], dtype=jnp.complex64)
    lambda5 = jnp.array([[0, 0, -1j],
                         [0, 0, 0],
                         [1j, 0, 0]], dtype=jnp.complex64)
    lambda6 = jnp.array([[0, 0, 0],
                         [0, 0, 1],
                         [0, 1, 0]], dtype=jnp.complex64)
    lambda7 = jnp.array([[0, 0, 0],
                         [0, 0, -1j],
                         [0, 1j, 0]], dtype=jnp.complex64)
    lambda8 = (1/jnp.sqrt(3)) * jnp.array([[1, 0, 0],
                                           [0, 0, 0],
                                           [0, 0, -2]], dtype=jnp.complex64)
    return {'1': lambda1, '2': lambda2, '3': lambda3, '4': lambda4,
            '5': lambda5, '6': lambda6, '7': lambda7, '8': lambda8}

def encode_qutrit(qutrit_state: jnp.ndarray, weights: dict):
    """
    Applies the encoder unitary U to an input qutrit state.
    
    The unitary is defined as:
        U = exp(i * (w1 * lambda1 + w2 * lambda2 + ... + w8 * lambda8))
    The goal is to optimize the weights such that the first element of the
    encoded state is (nearly) 0.
    
    Parameters:
        qutrit_state (jnp.ndarray): A 3-dimensional complex vector.
        weights (dict): Dictionary with keys '1' through '8' for tunable weights.
        
    Returns:
        tuple: (encoded_state, encoder_unitary) where:
            - encoded_state is the 3-dimensional encoded state,
            - encoder_unitary is the unitary matrix used for encoding.
    """
    generators = get_generators()
    # Sum weighted generators to form the effective generator
    generator = (weights['1'] * generators['1'] +
                 weights['2'] * generators['2'] +
                 weights['3'] * generators['3'] +
                 weights['4'] * generators['4'] +
                 weights['5'] * generators['5'] +
                 weights['6'] * generators['6'] +
                 weights['7'] * generators['7'] +
                 weights['8'] * generators['8'])
    U = expm(1j * generator)
    encoded_state = jnp.dot(U, qutrit_state)
    return encoded_state, U