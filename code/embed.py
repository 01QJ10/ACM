import numpy as np

def embed(rho_4x4: np.ndarray) -> np.ndarray:
    """
    Embeds a 4x4 density matrix (two qubits) into a 9x9 density matrix (two qutrits),
    such that the qubit basis |0>,|1> is mapped to the qutrit basis |1>,|2> respectively.
    
    Parameters:
        rho_4x4 (np.ndarray): A 4x4 density matrix.
        
    Returns:
        np.ndarray: A 9x9 embedded density matrix.
    """
    # Single-qubit to qutrit embedding operator (3x2)
    E = np.array([[0, 0],
                  [1, 0],
                  [0, 1]], dtype=complex)
    
    # Construct the 9x9 embedded density matrix using the embedding operator
    rho_9x9 = E @ rho_4x4 @ E.conjugate().T
    return rho_9x9

if __name__ == "__main__":
    # Example usage: create a random 4x4 density matrix from a pure state and embed it.
    np.random.seed(42)
    psi_4 = np.random.randn(4) + 1j * np.random.randn(4)
    psi_4 /= np.linalg.norm(psi_4)
    rho_4x4 = np.outer(psi_4, psi_4.conjugate())

    # Embed into two-qutrit space
    rho_9x9 = embed(rho_4x4)

    print("Original 4x4 density matrix:\n", rho_4x4)
    print("\nEmbedded 9x9 density matrix:\n", rho_9x9)
    print("\nTrace of rho_9x9 =", np.trace(rho_9x9))