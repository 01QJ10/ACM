import os
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

# Set random seed for reproducibility.
seed = 4
np.random.seed(seed)
torch.manual_seed(seed)

# Define a PyTorch Dataset to load qutrit states from a text file.
class QutritDataset(Dataset):
    def __init__(self, file_path):
        # Each row in the file is formatted as:
        # [Re(z0), Re(z1), Re(z2), Im(z0), Im(z1), Im(z2), overlap]
        data = np.loadtxt(file_path)
        # Reassemble into a complex vector for each state, ignoring the extra overlap column.
        self.states = data[:, :3] + 1j * data[:, 3:6]

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
      - An occupation loss term (beta * |encoded_state[0]|^2) to penalize population in the first element.
    
    When beta is 1.0, the total loss is equal to the occupation loss;
    otherwise, the total loss is equal to cloning_loss + beta * occupation_loss.
    """
    encoded_state, encoder_unitary = encode_qutrit(qutrit_state, weights)
    occupation_loss = jnp.abs(encoded_state[0])**2

    # Extract and normalize the effective 2-level state (elements 1 and 2).
    effective_part = encoded_state[1:3]
    norm_eff = jnp.linalg.norm(effective_part)
    effective_psi = jax.lax.cond(norm_eff > 0,
                                 lambda _: effective_part / norm_eff,
                                 lambda _: effective_part,
                                 operand=None)

    # Perform the cloning transformation on the effective state.
    rho_AB, rho_A, rho_B = buzek_hillery_clone(effective_psi)
    rho_A_embed = embed(rho_A)
    rho_B_embed = embed(rho_B)
    decoded_rho_A = decode_qubit_to_qutrit(rho_A_embed, encoder_unitary)
    decoded_rho_B = decode_qubit_to_qutrit(rho_B_embed, encoder_unitary)

    # jax.debug.print("Trace of decoded_rho_A: {}", jnp.trace(decoded_rho_A))
    # jax.debug.print("Trace of decoded_rho_B: {}", jnp.trace(decoded_rho_B))
    # Calculate the fidelities for both clones.
    F_A = fidelity(qutrit_state, decoded_rho_A)
    F_B = fidelity(qutrit_state, decoded_rho_B)
    cloning_loss = 1 - F_A - F_B + (F_A - F_B)**2

    # Use jax.lax.cond to choose the branch in a JIT-friendly manner.
    total_loss = jax.lax.cond(jnp.abs(beta - 1.0) < 1e-6,
                              lambda _: occupation_loss,
                              lambda _: cloning_loss + beta * occupation_loss,
                              operand=None)
    return total_loss, cloning_loss, F_A, F_B

def compute_loss_batch(weights, batch_qutrit_states, beta=1.0):
    """
    Compute the average loss over a batch of qutrit states.
    Uses jax.vmap to vectorize the computation of loss for each state.
    """
    loss_fn = jax.vmap(lambda state: compute_loss(weights, state, beta))
    losses = loss_fn(batch_qutrit_states)
    losses, cloning_losses, F_A, F_B = losses[0], losses[1], losses[2], losses[3]
    return jnp.mean(losses), jnp.mean(cloning_losses), jnp.mean(F_A), jnp.mean(F_B)

def total_loss_for_grad(weights, batch_qutrit_states, beta=1.0):
    total_loss, _, _, _ = compute_loss_batch(weights, batch_qutrit_states, beta)
    return total_loss

def train(dataloader, num_epochs=200, learning_rate=0.01, beta=1.0):
    # Get local devices; we assume here that there are 2 GPUs.
    devices = jax.local_devices()
    n_devices = len(devices)
    print(f"Using {n_devices} GPU(s): {devices}")

    # Initialize weights randomly.
    weights = {str(i): float(np.random.randn()) for i in range(1, 9)}
    # Replicate weights across devices.
    weights = jax.device_put_replicated(weights, devices)

    loss_history = []
    cloning_loss_history = []
    FA_history = []
    FB_history = []
    
    grad_fn = jax.grad(total_loss_for_grad, argnums=0)
    
    # Define update_step; average gradients over devices using pmean.
    def update_step(weights, batch, beta):
        grads = grad_fn(weights, batch, beta)
        grads = jax.lax.pmean(grads, axis_name='i')
        return jax.tree_util.tree_map(lambda w, g: w - learning_rate * g, weights, grads)
    
    # JIT-compile and parallelize update_step across devices.
    update_step_pmap = jax.pmap(update_step, static_broadcasted_argnums=(2,), axis_name='i')
    
    # Define a pmap version of compute_loss_batch.
    def loss_fn(weights, batch, beta):
        return compute_loss_batch(weights, batch, beta)
    loss_fn_pmap = jax.pmap(loss_fn, static_broadcasted_argnums=(2,), axis_name='i')
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_cloning_loss = 0.0
        total_FA = 0.0
        total_FB = 0.0
        n_batches = 0
        for batch in dataloader:
            if isinstance(batch, torch.Tensor):
                batch = batch.numpy()
            batch = jnp.array(batch)
            # Ensure batch is divisible by n_devices.
            B = batch.shape[0]
            new_B = (B // n_devices) * n_devices
            if new_B == 0:
                continue
            batch = batch[:new_B]
            # Reshape batch so that it is split across devices.
            batch = batch.reshape((n_devices, new_B // n_devices) + batch.shape[1:])
            
            # Compute loss on each shard.
            losses, cloning_losses, F_As, F_Bs = loss_fn_pmap(weights, batch, beta)
            avg_loss = jnp.mean(losses)
            avg_cloning_loss = jnp.mean(cloning_losses)
            avg_F_A = jnp.mean(F_As)
            avg_F_B = jnp.mean(F_Bs)
            
            epoch_loss += avg_loss
            epoch_cloning_loss += avg_cloning_loss
            total_FA += avg_F_A
            total_FB += avg_F_B
            n_batches += 1
            
            # Update weights in parallel.
            weights = update_step_pmap(weights, batch, beta)
        if n_batches > 0:
            avg_epoch_loss = epoch_loss / n_batches
            avg_epoch_cloning_loss = epoch_cloning_loss / n_batches
            avg_epoch_FA = total_FA / n_batches
            avg_epoch_FB = total_FB / n_batches
        else:
            avg_epoch_loss = 0.0
            avg_epoch_cloning_loss = 0.0
            avg_epoch_FA = 0.0
            avg_epoch_FB = 0.0
        
        loss_history.append(avg_epoch_loss)
        cloning_loss_history.append(avg_epoch_cloning_loss)
        FA_history.append(avg_epoch_FA)
        FB_history.append(avg_epoch_FB)
        print(f"Epoch {epoch}: Avg Loss = {avg_epoch_loss:.6f}")
        print(f"Epoch {epoch}: Avg Cloning Loss = {avg_epoch_cloning_loss:.6f}")
        print(f"Epoch {epoch}: Avg F_A = {avg_epoch_FA:.6f}, Avg F_B = {avg_epoch_FB:.6f}")
    
    # Retrieve a single copy of the final weights.
    final_weights = jax.tree_map(lambda x: x[0], weights)
    print("Final weights:", final_weights)
    return final_weights, loss_history, cloning_loss_history, FA_history, FB_history

def main():
    # Define a trial variable to easily rename each trial.
    trial = f"check_trial{seed}_noise"  # Change this string to rename the trial as needed.
    trial_dir = f"../results/{trial}"
    # os.makedirs(trial_dir, exist_ok=True)
    
    p_list = np.arange(0.01, 0.51, 0.01)
    beta_list = [8.0]
    lr_list = [0.01]
    num_epochs = 300
    batch_size = 100
    
    hyperparams_results = {}
    for count, p in enumerate(p_list):
        if count < 9:
            name = f"p_0_0{int(p * 100)}"
            print(f"Count: {count}, p: {p}, name: {name}")
        else:
            name = f"p_0_{int(p * 100)}"
            print(f"Count: {count}, p: {p}, name: {name}")
        dataset = QutritDataset(f"../noise/check/{seed}/{name}.txt")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for beta in beta_list:
            for lr in lr_list:
                print(f"Running trial: p = {p}, β = {beta}, lr = {lr}")
                weights, loss_history, cloning_loss_history, FA_history, FB_history = train(
                    dataloader, num_epochs=num_epochs, learning_rate=lr, beta=beta
                )
                # Save final encoder unitary.
                ref_state = dataset[0]
                _, encoder_unitary = encode_qutrit(ref_state, weights)
                # with open(f"{trial_dir}/U_p_{p}_beta_{beta}_lr_{lr}.txt", "w") as f:
                #     f.write(np.array2string(np.array(encoder_unitary), precision=6, separator=", "))
                # Save loss histories for this trial.
                hyperparams_results[(p, beta, lr)] = {
                    "loss_history": loss_history,
                    "cloning_loss_history": cloning_loss_history,
                    "FA_history": FA_history,
                    "FB_history": FB_history
                }
        # Save the hyperparameter results dictionary to a file.
        # np.save(f"{trial_dir}/{name}_results.npy", hyperparams_results)

if __name__ == "__main__":
    main()