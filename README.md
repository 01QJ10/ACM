# Quantum Autocloning Machine (ACM)

This repository implements a quantum autoencoder that compresses a qutrit into a qubit, applies the Buzek-Hillery cloning protocol on the qubit, and decodes it back into a qutrit. The main objective is to investigate whether operating in a lower-dimensional space can enhance cloning fidelity.

The implementation leverages JAX for numerical computations, NumPy for data handling, and PyTorch for dataset management. The project also includes utilities for generating qutrit states, introducing controlled noise, and visualizing the results.

## Repository Structure
```
quantum-autocloning-machine/
├── Code/
│   ├── embed.py                 # Embeds a 4x4 density matrix into a 9x9 matrix (two qutrits)
│   ├── encoder.py               # Implements the qutrit-to-qubit encoding via a unitary transformation
│   ├── decoder.py               # Decodes the cloned qubit(s) back into a qutrit state
│   ├── cloning.py               # Implements the simplified Buzek-Hillery cloning protocol
│   ├── loss.py                  # Contains fidelity and loss function definitions
│   ├── generate_qutrit_state.py # Generates different types of qutrit states (random, pseudo, almost-qubit)
│   ├── utils.py                 # Utility functions (loading states, computing unitary metrics, etc.)
│   ├── noise.py                 # Generates qutrit states with controlled noise/overlap parameters
│   ├── main.py                  # Single-device training pipeline (encoding, cloning, decoding, loss evaluation)
│   ├── main2.py                 # Multi-device (pmap) training pipeline with a full processing pipeline
│   └── plot.py                  # Plots and visualizes training results and fidelity metrics
├── tests/
│   ├── test_encoder.py          # Unit tests for the encoder module
│   ├── test_decoder.py          # Unit tests for the decoder module
│   ├── test_cloning.py          # Unit tests for the cloning module
│   └── test_loss.py             # Unit tests for the loss function module
├── results/                     # Directory for training output, plots, and saved results
├── data/                        # Input data files (e.g., qutrit state datasets)
└── .gitignore                   # Git ignore file to exclude unnecessary files
```

## Quick Start

1. **Generate Qutrit States:**  
   Use the `generate_qutrit_state.py` module to create training and test datasets. For example:
   ```bash
    python Code/generate_qutrit_state.py --mode random --num_states 5000 --test_size 0.2
    ```

2.	Training:
    Run main.py or main2.py to start the training process with your chosen configuration.
```bash
python Code/main.py
```

3.	Evaluation and Plotting:
    After training, review results in the results/ directory and use plot.py to visualize metrics.

Dependencies
	•	Python 3.x
	•	JAX
	•	NumPy
	•	PyTorch
	•	scikit-learn
	•	Matplotlib

For further details and configuration options, please refer to the individual module docstrings.

Happy cloning and encoding!