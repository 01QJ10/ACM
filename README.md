# QAM
# Quantum Autoencoder with Buzek-Hillery Cloning

This repository implements a quantum autoencoder that encodes a qutrit into a qubit, applies the Buzek-Hillery cloning protocol to the qubit, and decodes it back to a qutrit. The objective is to explore whether operating in a lower-dimensional space can enhance cloning fidelity.

## Repository Structure
```
quantum-autoencoder/
├── Code/
│   ├── encoder.py         # Implements the qutrit-to-qubit encoding via a unitary transformation
│   ├── decoder.py         # Decodes the cloned qubit(s) back into a qutrit state
│   ├── cloning.py         # Implements a simplified Buzek-Hillery cloning protocol
│   ├── loss.py            # Contains the fidelity and loss function definitions
│   └── main.py            # Ties together encoding, cloning, decoding, and loss evaluation
├── Docs/
│   ├── README.md          # Overview, installation, and usage instructions
│   └── QuantumMechanicsBackground.md  # Theoretical background and mathematical derivations
├── Tests/
│   ├── test_encoder.py    # Unit tests for the encoder module
│   ├── test_decoder.py    # Unit tests for the decoder module
│   ├── test_cloning.py    # Unit tests for the cloning module
│   └── test_loss.py       # Unit tests for the loss function module
└── Examples/
    └── run_example.py     # Example script demonstrating the overall process
```