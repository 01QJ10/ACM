---

### Docs/QuantumMechanicsBackground.md

```markdown
# Quantum Mechanics Background and Mathematical Derivations

## Overview

This document provides the theoretical background for the quantum autoencoder project, including key quantum mechanics principles, the encoding/decoding operations, and the Buzek-Hillery cloning protocol.

## Qutrit and Qubit States

- **Qutrit:** A quantum system with a three-dimensional Hilbert space.
- **Qubit:** A quantum system with a two-dimensional Hilbert space.

In this project, a qutrit is encoded into a qubit by projecting onto a two-dimensional subspace followed by a unitary rotation.

## Encoding via Unitary Transformation

The encoding unitary is defined as:
\[
U = \exp\left(i\sum_{j=x,y,z} w_j \sigma_j\right)
\]
where the \(\sigma_j\) are the Pauli matrices:
- \(\sigma_x = \begin{pmatrix}0 & 1\\1 & 0\end{pmatrix}\)
- \(\sigma_y = \begin{pmatrix}0 & -i\\ i & 0\end{pmatrix}\)
- \(\sigma_z = \begin{pmatrix}1 & 0\\0 & -1\end{pmatrix}\)

The weights \(w_j\) are tunable parameters that can be optimized for better fidelity.

## Buzek-Hillery Cloning Protocol

The Buzek-Hillery cloning machine is a universal quantum cloning protocol with an optimal cloning fidelity of approximately \( \frac{5}{6} \) for qubits. In this project, the cloning process is simulated by combining the input qubit with its orthogonal complement.

## Decoding Process

The decoding operation reconstructs the qutrit state from the cloned qubit(s). It is implemented as:
\[
U_{\text{decoder}} = U^\dagger \otimes U^\dagger
\]
After applying this unitary, a projection onto a 3-dimensional subspace retrieves the qutrit state.

## Loss Function

The loss function used to optimize the cloning process is defined as:
\[
L = 1 - F_A - F_B + (F_A - F_B)^2
\]
where \(F_A\) and \(F_B\) are the fidelities of the two decoded clones with respect to the original state:
\[
F(\psi,\phi) = |\langle \psi | \phi \rangle|^2
\]

## Future Work

Future improvements may include:
- Gradient-based optimization of the encoder weights.
- Exploration of alternative encoding/decoding strategies.
- Enhanced simulation of the Buzek-Hillery cloning protocol.