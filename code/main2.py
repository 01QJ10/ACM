import os
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from encoder import encode_qutrit
from cloning import buzek_hillery_clone
from loss import fidelity
from decoder import decode_qubit_to_qutrit
from embed import embed

seed = 5
np.random.seed(seed)
torch.manual_seed(seed)

class QutritDataset(Dataset):
    """
    PyTorch Dataset for loading qutrit states from a text file.
    """
    def __init__(self, file_path: str):
        data = np.loadtxt(file_path)
        self.states = data[:, :3] + 1j * data[:, 3:6]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.states[idx]

def run_pipeline(qutrit_state: jnp.ndarray, weights):
    """
    Runs the full pipeline (encoding -> cloning -> decoding) on a qutrit state.
    
    Returns the decoded density matrix.
    """
    encoded_state, encoder_unitary = encode_qutrit(qutrit_state, weights)
    effective_part = encoded_state[1:3]
    norm_eff = jnp.linalg.norm(effective_part)
    effective_psi = jax.lax.cond(norm_eff > 0,
                                 lambda _: effective_part / norm_eff,
                                 lambda _: effective_part,
                                 operand=None)
    rho_AB, rho_A, rho_B = buzek_hillery_clone(effective_psi)
    rho_A_embed = embed(rho_A)
    decoded_rho = decode_qubit_to_qutrit(rho_A_embed, encoder_unitary)
    return decoded_rho

def compute_loss(weights, qutrit_state, beta: float = 1.0):
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
    loss_fn = jax.vmap(lambda state: compute_loss(weights, state, beta))
    losses = loss_fn(batch_qutrit_states)
    total_loss, cloning_losses, F_A, F_B = losses
    return jnp.mean(total_loss), jnp.mean(cloning_losses), jnp.mean(F_A), jnp.mean(F_B)

def total_loss_for_grad(weights, batch_qutrit_states, beta: float = 1.0):
    total_loss, _, _, _ = compute_loss_batch(weights, batch_qutrit_states, beta)
    return total_loss

def train(dataloader, ref_state, num_epochs: int = 200, learning_rate: float = 0.01, beta: float = 1.0):
    devices = jax.local_devices()
    n_devices = len(devices)
    print(f"Using {n_devices} device(s): {devices}")

    weights = {str(i): float(np.random.randn()) for i in range(1, 9)}
    weights = jax.device_put_replicated(weights, devices)

    loss_history = []
    cloning_loss_history = []
    FA_history = []
    FB_history = []
    
    grad_fn = jax.grad(total_loss_for_grad, argnums=0)
    
    def update_step(weights, batch, beta):
        grads = grad_fn(weights, batch, beta)
        grads = jax.lax.pmean(grads, axis_name='i')
        return jax.tree_util.tree_map(lambda w, g: w - learning_rate * g, weights, grads)
    
    update_step_pmap = jax.pmap(update_step, static_broadcasted_argnums=(2,), axis_name='i')
    loss_fn_pmap = jax.pmap(lambda w, b, beta: compute_loss_batch(w, b, beta),
                            static_broadcasted_argnums=(2,), axis_name='i')
    
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
            B = batch.shape[0]
            new_B = (B // n_devices) * n_devices
            if new_B == 0:
                continue
            batch = batch[:new_B].reshape((n_devices, new_B // n_devices) + batch.shape[1:])
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
            weights = update_step_pmap(weights, batch, beta)
        if n_batches > 0:
            avg_epoch_loss = epoch_loss / n_batches
            avg_epoch_cloning_loss = epoch_cloning_loss / n_batches
            avg_epoch_FA = total_FA / n_batches
            avg_epoch_FB = total_FB / n_batches
        else:
            avg_epoch_loss = avg_epoch_cloning_loss = avg_epoch_FA = avg_epoch_FB = 0.0
        
        loss_history.append(avg_epoch_loss)
        cloning_loss_history.append(avg_epoch_cloning_loss)
        FA_history.append(avg_epoch_FA)
        FB_history.append(avg_epoch_FB)
        print(f"Epoch {epoch}: Avg Loss = {avg_epoch_loss:.6f}")
        print(f"Epoch {epoch}: Avg Cloning Loss = {avg_epoch_cloning_loss:.6f}")
        print(f"Epoch {epoch}: Avg F_A = {avg_epoch_FA:.6f}, Avg F_B = {avg_epoch_FB:.6f}")
    
    final_weights = jax.tree_map(lambda x: x[0], weights)
    state_eff = jnp.array([0, ref_state[1], ref_state[2]], dtype=jnp.complex64)
    decoded_rho_eff = run_pipeline(state_eff, final_weights)
    F_cl_rho = fidelity(state_eff, decoded_rho_eff)
    state_occ = jnp.array([ref_state[0], 0, 0], dtype=jnp.complex64)
    decoded_rho_occ = run_pipeline(state_occ, final_weights)
    F_cl_rho0 = fidelity(state_occ, decoded_rho_occ)
    
    print("Final weights:", final_weights)
    return final_weights, loss_history, cloning_loss_history, FA_history, FB_history, F_cl_rho, F_cl_rho0

def main():
    trial = f"check_trial{seed}_noise"
    trial_dir = f"../results/{trial}"
    os.makedirs(trial_dir, exist_ok=True)
    
    p_list = np.arange(0.01, 1.01, 0.01)
    beta_list = [8.0]
    lr_list = [0.01]
    num_epochs = 300
    batch_size = 100
    hyperparams_results = {}

    for count, p in enumerate(p_list):
        name = f"p_0_{int(p * 100):02d}" if p < 0.1 else f"p_0_{int(p * 100)}"
        print(f"Processing: p = {p}, name = {name}")
        dataset = QutritDataset(f"../noise/check/{seed}/{name}.txt")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        ref_state = dataset[0]
        for beta in beta_list:
            for lr in lr_list:
                print(f"Running trial: p = {p}, β = {beta}, lr = {lr}")
                weights, loss_hist, cloning_loss_hist, FA_hist, FB_hist, F_cl_rho, F_cl_rho0 = train(
                    dataloader, ref_state, num_epochs=num_epochs, learning_rate=lr, beta=beta
                )
                F_approx = (1 - p) * F_cl_rho + p * F_cl_rho0
                hyperparams_results[(p, beta, lr)] = {
                    "loss_history": loss_hist,
                    "cloning_loss_history": cloning_loss_hist,
                    "FA_history": FA_hist,
                    "FB_history": FB_hist,
                    "F_cl_rho": float(F_cl_rho),
                    "F_cl_rho0": float(F_cl_rho0),
                    "F_approx": float(F_approx)
                }
        np.save(f"{trial_dir}/hyperparams_results.npy", hyperparams_results)

if __name__ == "__main__":
    main()