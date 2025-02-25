# main.py
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from encoder import encode_qutrit  # applies the unitary to a qutrit state
from cloning import buzek_hillery_clone  # performs the cloning transformation
from loss import fidelity  # computes fidelity ⟨ψ|ρ|ψ⟩
from decoder import decode_qubit_to_qutrit  # decodes a qubit state to a qutrit state
from embed import embed  # embeds a qutrit state into a qubit state
# Define a PyTorch Dataset to load qutrit states from a text file.
class QutritDataset(Dataset):
    def __init__(self, file_path):
        # Each row in the file is formatted as:
        # [Re(z0), Re(z1), Re(z2), Im(z0), Im(z1), Im(z2)]
        data = np.loadtxt(file_path)
        # Reassemble into a complex vector for each state.
        self.states = data[:1000, :3] + 1j * data[:1000, 3:]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # Return a single qutrit state as a numpy array.
        return self.states[idx]
    
    def test(self):
        return None

def compute_loss(weights, qutrit_state, beta=1.0):
    """
    Compute the loss for one qutrit state.
    
    The loss consists of:
      - A cloning loss based on the fidelities of the two cloned outputs.
      - An occupation loss term (beta * |encoded_state[0]|^2) to penalize population
        in the first element (i.e. to encourage encoding into the lower two levels).
    
    Parameters:
      weights: dictionary with keys '1', '2', '3' (encoder parameters)
      qutrit_state: a 3D complex state (jnp.ndarray)
      beta: weight for the occupation loss term.
    
    Returns:
      Total loss (scalar)
    """
    # Apply the encoder unitary to the full qutrit state.
    # print(f"qutrit shape: {qutrit_state.shape}")
    encoded_state, encoder_unitary = encode_qutrit(qutrit_state, weights)
    occupation_loss = jnp.abs(encoded_state[0])**2

    # Extract the effective 2-level state (elements 1 and 2) and normalize.
    effective_part = encoded_state[1:3]
    norm_eff = jnp.linalg.norm(effective_part)
    effective_psi = jax.lax.cond(norm_eff > 0, lambda _: effective_part / norm_eff, lambda _: effective_part, operand=None)

    # Perform the cloning transformation on the effective state.
    rho_AB, rho_A, rho_B = buzek_hillery_clone(effective_psi)
    rho_A_embed = embed(rho_A)
    rho_B_embed = embed(rho_B)
    # print(rho_B.shape)
    decoded_rho_A = decode_qubit_to_qutrit(rho_A_embed, encoder_unitary)
    decoded_rho_B = decode_qubit_to_qutrit(rho_B_embed, encoder_unitary)

    # Calculate the fidelities for both clones.
    # F_A = fidelity(qutrit_state, decoded_rho_A)
    # F_B = fidelity(qutrit_state, decoded_rho_B)
    F_A = fidelity(qutrit_state[1:3], rho_A)
    F_B = fidelity(qutrit_state[1:3], rho_B)
    # print(f"Fidelities: F_A = {F_A}, F_B = {F_B}")
    cloning_loss = 1 - F_A - F_B + (F_A - F_B)**2
    total_loss = cloning_loss + beta * occupation_loss
    # print(f"Total loss: {total_loss}")
    return total_loss

def compute_loss_batch(weights, batch_qutrit_states, beta=1.0):
    """
    Compute the average loss over a batch of qutrit states.
    
    Uses jax.vmap to vectorize the computation of loss for each state.
    """
    loss_fn = jax.vmap(lambda state: compute_loss(weights, state, beta))
    losses = loss_fn(batch_qutrit_states)
    return jnp.mean(losses)

def train(dataloader, num_epochs=50, learning_rate=0.01, beta=1.0):
    # Initialize encoder weights (for keys '1', '2', '3').
    weights = {'1': 0.3, '2': 0.2, '3': 0.1, '4': 0.2, '5': 0.5, '6': 0.1, '7': 0.2, '8': 0.3}
    loss_history = []

    # Define the gradient function (w.r.t. weights) of the batch loss.
    grad_fn = jax.grad(compute_loss_batch, argnums=0)

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch in dataloader:
            # 'batch' is a numpy array (or PyTorch tensor) of shape (batch_size, 3).
            # If it's a PyTorch tensor, convert to numpy.
            if isinstance(batch, torch.Tensor):
                batch = batch.numpy()
            # Convert to a JAX array.
            batch = jnp.array(batch)
            # Compute the average loss for this batch.
            batch_loss = compute_loss_batch(weights, batch, beta)
            epoch_loss += batch_loss
            num_batches += 1

            # Compute gradients and update weights.
            grads = grad_fn(weights, batch, beta)
            weights = jax.tree_map(lambda w, g: w - learning_rate * g, weights, grads)
        # print(epoch_loss)
        # print(num_batches)
        avg_epoch_loss = epoch_loss / num_batches
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch}: Avg Loss = {avg_epoch_loss:.6f}, Weights = {weights}")

    print("Final weights:", weights)
    return weights, loss_history

def main():
    # Create a PyTorch DataLoader for the training qutrit states.
    dataset = QutritDataset("data/train.txt")
    dataloader = DataLoader(dataset, batch_size=30, shuffle=True)

    # Batch size is the number of qutrit states in each batch.
    weights, loss_history = train(dataloader, num_epochs=50, learning_rate=0.1, beta=0.0)

    # Plot the loss
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss")
    plt.grid(alpha=0.35)
    plt.show()

if __name__ == "__main__":
    main()