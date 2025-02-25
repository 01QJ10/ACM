import numpy as np

def embed(rho_4x4: np.ndarray) -> np.ndarray:
    """
    Embed a 4x4 density matrix (two qubits) into a 9x9 matrix (two qutrits),
    such that each qubit basis |0>,|1> goes to qutrit basis |1>,|2> respectively.
    """
    # Single-qubit->qutrit embedding operator (3x2)
    E = np.array([[0, 0],
                  [1, 0],
                  [0, 1]], dtype=complex)
    
    # Two-qubit->two-qutrit embedding (9x4)
    # E_tensor = np.kron(E, E)  # shape: (9,4)
    
    # Construct the 9x9 embedded density matrix
    rho_9x9 = E @ rho_4x4 @ E.conjugate().T
    return rho_9x9

# --- Example usage ---
if __name__ == "__main__":
    # Example: a random 4x4 density matrix (two qubits)
    # For demonstration, let's build one from a random pure state:
    np.random.seed(42)
    psi_4 = np.random.randn(4) + 1j*np.random.randn(4)
    psi_4 /= np.linalg.norm(psi_4)
    rho_4x4 = np.outer(psi_4, psi_4.conjugate())

    # Embed into two-qutrit space
    rho_9x9 = embed_2qubits_into_2qutrits(rho_4x4)

    print("Original 4x4 density matrix:\n", rho_4x4)
    print("\nEmbedded 9x9 density matrix:\n", rho_9x9)
    print("\nTrace of rho_9x9 = ", np.trace(rho_9x9))