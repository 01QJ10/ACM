import jax.numpy as jnp
import numpy as np


def qubit_to_qutrit_isometry() -> np.ndarray:
    """Map qubit basis |0>, |1> into qutrit basis |1>, |2>."""
    return jnp.array([[0, 0],
                      [1, 0],
                      [0, 1]], dtype=jnp.complex64)


def embed_single_qubit_density(rho_2x2: np.ndarray) -> np.ndarray:
    """Embed a one-clone 2x2 qubit density matrix into the qutrit space."""
    if rho_2x2.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 density matrix, got {rho_2x2.shape}")
    E = qubit_to_qutrit_isometry()
    return E @ rho_2x2 @ E.conjugate().T


def embed_two_qubit_density(rho_4x4: np.ndarray) -> np.ndarray:
    """Embed a joint two-qubit 4x4 density matrix into the two-qutrit 9x9 space."""
    if rho_4x4.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 density matrix, got {rho_4x4.shape}")
    E = qubit_to_qutrit_isometry()
    E2 = jnp.kron(E, E)
    return E2 @ rho_4x4 @ E2.conjugate().T


def embed(rho: np.ndarray) -> np.ndarray:
    """
    Backwards-compatible embedding helper.

    A 2x2 reduced clone is embedded into a one-qutrit 3x3 density matrix.
    A 4x4 joint clone state is embedded into a two-qutrit 9x9 density matrix.
    """
    if rho.shape == (2, 2):
        return embed_single_qubit_density(rho)
    if rho.shape == (4, 4):
        return embed_two_qubit_density(rho)
    raise ValueError(f"Expected a 2x2 or 4x4 density matrix, got {rho.shape}")

if __name__ == "__main__":
    # Example usage: create a random 4x4 density matrix from a pure state and embed it.
    np.random.seed(42)
    psi_4 = np.random.randn(4) + 1j * np.random.randn(4)
    psi_4 /= np.linalg.norm(psi_4)
    rho_4x4 = np.outer(psi_4, psi_4.conjugate())

    # Embed into two-qutrit space
    rho_9x9 = embed_two_qubit_density(rho_4x4)

    print("Original 4x4 density matrix:\n", rho_4x4)
    print("\nEmbedded 9x9 density matrix:\n", rho_9x9)
    print("\nTrace of rho_9x9 =", np.trace(rho_9x9))
