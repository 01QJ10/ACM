import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

def get_generators():
    """
    Returns three 3x3 generators (analogous to Pauli matrices for a qutrit)
    using a subset of the Gell-Mann matrices:
    
    lambda1 = [[0, 1, 0],
               [1, 0, 0],
               [0, 0, 0]]
               
    lambda2 = [[0, -i, 0],
               [i,  0, 0],
               [0,  0, 0]]
               
    lambda3 = [[1,  0, 0],
               [0, -1, 0],
               [0,  0, 0]]
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
    return {'1': lambda1, '2': lambda2, '3': lambda3, '4': lambda4, '5': lambda5, '6': lambda6, 
            '7': lambda7, '8': lambda8}

def encode_qutrit(qutrit_state, weights):
    """
    Applies the encoder unitary U to the input qutrit state.
    
    The unitary is defined as:
        U = exp(i * (w1 * lambda1 + w2 * lambda2 + w3 * lambda3))
    The aim is to optimize the weights such that the first element of the
    encoded state becomes (nearly) 0.
    
    Parameters:
        qutrit_state (jnp.ndarray): A 3-dimensional complex vector.
        weights (dict): Dictionary with keys '1', '2', '3' for tunable weights.
        
    Returns:
        jnp.ndarray: The encoded qutrit state (3-dimensional complex vector).
    """
    generators = get_generators()
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