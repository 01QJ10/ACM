# Quantum Autocloning Machine (ACM)

This repository implements a quantum autoencoder that compresses a qutrit into a qubit, applies the Buzek-Hillery cloning protocol on the qubit, and decodes it back into a qutrit. The main objective is to investigate whether operating in a lower-dimensional space can enhance cloning fidelity.

The implementation leverages JAX for numerical computations, NumPy for data handling, and PyTorch for dataset management. The project also includes utilities for generating qutrit states, introducing controlled noise, and visualizing the results.

> **Note:** The `results/` and `noise/` folders are not included in the repository since all outputs and noise data can be generated from the code.

## Repository Structure

```bash
ACM/
├── code/
│   ├── embed.py                 # Embeds one- and two-clone qubit density matrices into qutrit spaces
│   ├── encoder.py               # Implements the qutrit-to-qubit encoding via a unitary transformation
│   ├── decoder.py               # Decodes the cloned qubit(s) back into a qutrit state
│   ├── cloning.py               # Implements the simplified Buzek-Hillery cloning protocol
│   ├── loss.py                  # Contains fidelity and loss function definitions
│   ├── generate_qutrit_state.py # Generates different types of qutrit states (random, pseudo, almost-qubit)
│   ├── utils.py                 # Utility functions (loading states, computing unitary metrics, etc.)
│   ├── noise.py                 # Generates qutrit states with controlled noise/overlap parameters
│   ├── main.py                  # Single-device training pipeline (encoding, cloning, decoding, loss evaluation)
│   ├── main2.py                 # Leakage-noise training pipeline with configurable seeds and paths
│   ├── plot.py                  # Plots and visualizes training results and fidelity metrics
│   ├── reviewer_response_experiments.py # Reviewer-response baselines and repeated-seed runs
│   └── train.pbs                # Example job script for HPC batch submission
├── tests/
│   ├── test_encoder.py          # Unit tests for the encoder module
│   ├── test_decoder.py          # Unit tests for the decoder module
│   ├── test_cloning.py          # Unit tests for the cloning module
│   ├── test_loss.py             # Unit tests for the loss function module
│   └── test_reviewer_response_experiments.py # Reviewer-response diagnostic tests
├── data/                        # Input data files (e.g., qutrit state datasets)
└── .gitignore                   # Git ignore file to exclude unnecessary files
```

## Quick Start

1. **Install Dependencies:**

   First, install all required dependencies from the provided `code/requirements.txt` file.

   - **Locally:**  
     Create and activate a virtual environment (optional but recommended), then run:
     ```bash
     pip install -r code/requirements.txt
     ```

   - **On an HPC System:**  
     Ensure your environment is set up with the necessary modules or virtual environment. The repository includes a sample job script (`code/train.pbs`) for batch submission. Adjust the script as needed for your HPC scheduler and submit the job:
     ```bash
     qsub code/train.pbs
     ```

2. **Generate Qutrit States:**  
   Use the `generate_qutrit_state.py` module to create training and test datasets. For example:
   ```bash
   python code/generate_qutrit_state.py --mode random --num_states 5000 --test_size 0.2
   ```
   The manuscript experiments use the checked-in `data/pseudo_train_*.txt` and
   `data/pseudo_test_*.txt` files for the structured qutrit ensembles.

3.	Training:
	- **Locally:** 
    Run `main.py` or `main2.py` to start the training process with your chosen configuration.
    ```bash
    python code/main.py
    ```

	- **On an HPC System:** 
    Submit the provided job script (`code/train.pbs`) or create a custom job script to run one of the training scripts.

4.	Evaluation and Plotting:
    After training, review the generated results in the `results/` directory. Use `plot.py` to visualize training metrics:
    ```bash
    python code/plot.py
    ```

## Reviewer-Response Reproduction

The reviewer-response experiments for baselines, repeated-seed smoke checks, and
noise-path smoke checks are collected in one deterministic entrypoint:

```bash
cd /Users/behlari/Desktop/FYP/QAM
python code/reviewer_response_experiments.py --mode all --output-dir results/reviewer_response
```

Useful narrower commands:

```bash
python code/reviewer_response_experiments.py --mode baselines --output-dir results/reviewer_response
python code/reviewer_response_experiments.py --mode smoke --epochs 5 --seeds 0 1 2 --output-dir results/reviewer_response
python code/reviewer_response_experiments.py --mode noise-smoke --seeds 0 1 2 --output-dir results/reviewer_response
```

The full repeated-seed reviewer-response run used for the Entropy response can
be reproduced with:

```bash
python code/reviewer_response_experiments.py \
  --mode smoke \
  --epochs 200 \
  --seeds 0 1 2 3 4 \
  --learning-rate 0.01 \
  --beta 8.0 \
  --batch-size 50 \
  --max-train-states 2000 \
  --max-test-states 500 \
  --output-dir results/reviewer_response/full_beta8_lr001_seed0-4_epoch200
```

The noise data generator and noise training script now share the same default
seed/path convention:

```bash
python code/noise.py --seed 5 --output-root noise/check
python code/main2.py --seed 5 --noise-root noise/check --output-root results
python code/noise.py --seed 6 --output-root noise/check
python code/main2.py --seed 6 --noise-root noise/check --output-root results
python code/plot.py --seed-noise
```

Full multi-seed training remains more expensive than the smoke checks above;
record the exact seed list, epochs, and output directory when running it.

## Tests

Run the unit tests from the repository root:

```bash
python -m unittest discover -s tests
```


Dependencies
	•	Python 3.x
	•	JAX
	•	NumPy
	•	PyTorch
	•	scikit-learn
	•	Matplotlib

For further details and configuration options, please refer to the individual module docstrings. If you have further queries, kindly drop an email to bqjun0000@gmail.com to clarify :) !

Happy cloning and encoding!

**Citation:**
If you use this code in your work, please cite it as follows:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15160624.svg)](https://doi.org/10.5281/zenodo.15160624)
