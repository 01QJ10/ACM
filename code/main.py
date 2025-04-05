import os
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from encoder import encode_qutrit        # Applies the encoding unitary to a qutrit state.
from cloning import buzek_hillery_clone  # Performs the cloning transformation.
from loss import fidelity                # Computes fidelity ⟨ψ|ρ|ψ⟩.
from decoder import decode_qubit_to_qutrit# Decodes a qubit state to a qutrit state.
from embed import embed                  # Embeds a qubit state into a qutrit space.

# Set random seeds for reproducibility.
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

class QutritDataset(Dataset):
    """
    PyTorch Dataset for loading qutrit states from a text file.
    
    Each row in the file should be formatted as:
      [Re(z0), Re(z1), Re(z2), Im(z0), Im(z1), Im(z2), overlap]
    """
    def __init__(self, file_path: str):
        data = np.loadtxt(file_path)
        self.states = data[:, :3] + 1j * data[:, 3:6]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.states[idx]

def compute_loss(weights, qutrit_state, beta: float = 1.0):
    """
    Compute the loss for a single qutrit state.
    
    The loss comprises:
      - A cloning loss (based on fidelities of the two clones).
      - An occupation loss (penalizing population in the first element).
    
    Args:
        weights: Encoder weights.
        qutrit_state: Input qutrit state.
        beta: Scaling factor for the occupation loss.
    
    Returns:
        A tuple (total_loss, cloning_loss, F_A, F_B).
    """
    encoded_state, encoder_unitary = encode_qutrit(qutrit_state, weights)
    occupation_loss = jnp.abs(encoded_state[0]) ** 2

    effective_part = encoded_state[1:3]
    norm_eff = jnp.linalg.norm(effective_part)
    effective_psi = jax.lax.cond(norm_eff > 0,
                                 lambda _: effective_part / norm_eff,
                                 lambda _: effective_part,
                                 operand=None)

    rho_AB, rho_A, rho_B = buzek_hillery_clone(effective_psi)
    rho_A_embed = embed(rho_A)
    rho_B_embed = embed(rho_B)
    decoded_rho_A = decode_qubit_to_qutrit(rho_A_embed, encoder_unitary)
    decoded_rho_B = decode_qubit_to_qutrit(rho_B_embed, encoder_unitary)

    F_A = fidelity(qutrit_state, decoded_rho_A)
    F_B = fidelity(qutrit_state, decoded_rho_B)
    cloning_loss = 1 - F_A - F_B + (F_A - F_B) ** 2

    total_loss = jax.lax.cond(jnp.abs(beta - 1.0) < 1e-6,
                              lambda _: occupation_loss,
                              lambda _: cloning_loss + beta * occupation_loss,
                              operand=None)
    return total_loss, cloning_loss, F_A, F_B

def compute_loss_batch(weights, batch_qutrit_states, beta: float = 1.0):
    """
    Compute the average loss over a batch of qutrit states.
    
    Uses jax.vmap to vectorize loss computation.
    """
    loss_fn = jax.vmap(lambda state: compute_loss(weights, state, beta))
    losses = loss_fn(batch_qutrit_states)
    total_loss, cloning_losses, F_A, F_B = losses
    return jnp.mean(total_loss), jnp.mean(cloning_losses), jnp.mean(F_A), jnp.mean(F_B)

def total_loss_for_grad(weights, batch_qutrit_states, beta: float = 1.0):
    total_loss, _, _, _ = compute_loss_batch(weights, batch_qutrit_states, beta)
    return total_loss

def train(dataloader, num_epochs: int = 200, learning_rate: float = 0.01, beta: float = 1.0):
    # Initialize weights randomly.
    weights = {str(i): float(np.random.randn()) for i in range(1, 9)}
    loss_history = []
    cloning_loss_history = []
    FA_history = []
    FB_history = []
    grad_fn = jax.grad(total_loss_for_grad, argnums=0)

    @jax.jit
    def update_step(weights, batch, beta):
        grads = grad_fn(weights, batch, beta)
        return jax.tree_map(lambda w, g: w - learning_rate * g, weights, grads)

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_cloning_loss = 0
        total_FA = 0
        total_FB = 0
        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, torch.Tensor):
                batch = batch.numpy()
            batch = jnp.array(batch)
            batch_loss, cloning_loss, F_A, F_B = compute_loss_batch(weights, batch, beta)
            epoch_loss += batch_loss
            epoch_cloning_loss += cloning_loss
            total_FA += F_A
            total_FB += F_B
            num_batches += 1
            weights = update_step(weights, batch, beta)
        
        avg_loss = epoch_loss / num_batches
        avg_cloning_loss = epoch_cloning_loss / num_batches
        avg_FA = total_FA / num_batches
        avg_FB = total_FB / num_batches

        loss_history.append(avg_loss)
        cloning_loss_history.append(avg_cloning_loss)
        FA_history.append(avg_FA)
        FB_history.append(avg_FB)
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.6f}, Weights = {weights}")
        print(f"Epoch {epoch}: Avg Cloning Loss = {avg_cloning_loss:.6f}")
        print(f"Epoch {epoch}: Avg F_A = {avg_FA:.6f}, Avg F_B = {avg_FB:.6f}")

    print("Final weights:", weights)
    return weights, loss_history, cloning_loss_history, FA_history, FB_history

def main():
    trial = "trial1"
    trial_dir = f"../results/{trial}"
    os.makedirs(trial_dir, exist_ok=True)
    
    # Hyperparameters.
    chi_list = [0.85, 0.9, 0.95, 0.99]
    beta_list = [1.0, 8.0]
    lr_list = [0.01, 0.001]
    num_epochs = 200
    batch_size = 100
    
    hyperparams_results = {}
    for chi in chi_list:
        chi_str = str(int(chi * 100))
        dataset = QutritDataset(f"../data/pseudo_train_{chi_str}.txt")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for beta in beta_list:
            for lr in lr_list:
                print(f"Running trial: χ = {chi}, β = {beta}, lr = {lr}")
                weights, loss_hist, cloning_loss_hist, FA_hist, FB_hist = train(
                    dataloader, num_epochs=num_epochs, learning_rate=lr, beta=beta
                )
                ref_state = dataset[0]
                _, encoder_unitary = encode_qutrit(ref_state, weights)
                with open(f"{trial_dir}/U_chi_{chi}_beta_{beta}_lr_{lr}.txt", "w") as f:
                    f.write(np.array2string(np.array(encoder_unitary), precision=6, separator=", "))
                plt.figure(figsize=(8, 6))
                plt.plot(loss_hist, label="Total Loss")
                plt.plot(cloning_loss_hist, label="Cloning Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Average Loss")
                plt.legend()
                plt.title(f"Training Loss, β = {beta}, lr = {lr}, χ = {chi}")
                plt.grid(alpha=0.35)
                plt.savefig(f"{trial_dir}/loss_chi_{chi}_beta_{beta}_lr_{lr}.png")
                plt.close()
                plt.figure(figsize=(8, 6))
                plt.plot(FA_hist, label="F_A")
                plt.plot(FB_hist, label="F_B")
                plt.xlabel("Epoch")
                plt.ylabel("Average Fidelity")
                plt.legend()
                plt.title(f"Average Fidelities, β = {beta}, lr = {lr}, χ = {chi}")
                plt.grid(alpha=0.35)
                plt.savefig(f"{trial_dir}/fidelity_chi_{chi}_beta_{beta}_lr_{lr}.png")
                plt.close()
                hyperparams_results[(chi, beta, lr)] = {
                    "loss_history": loss_hist,
                    "cloning_loss_history": cloning_loss_hist,
                    "FA_history": FA_hist,
                    "FB_history": FB_hist
                }
    np.save(f"{trial_dir}/hyperparams_results.npy", hyperparams_results)

if __name__ == "__main__":
    main()