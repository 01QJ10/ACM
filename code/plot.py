import numpy as np
import matplotlib.pyplot as plt

# # Load the results dictionary from the .npy file.
# results = np.load("../results/trial1/hyperparams_results.npy", allow_pickle=True).item()

# # Define color mapping for chi and marker mapping for lr and beta.
# colors = {0.99: ['#b8cdab', 'solid'], 
#           0.95: ['#004343', 'dotted'],  
#           0.90: ['#e5c185', 'dashed'], 
#           0.85: ['#c7522a', 'dashdot']}


# import re

# def extract_chi_value(label):
#     # This regex searches for one or more digits and optional decimal
#     # at the end of the string. Example matches: "0.85", "1.00", etc.
#     match = re.search(r'([\d.]+)$', label)
#     if match:
#         return float(match.group(1))
#     return None

# def sort_and_set_legend(ax):
#     handles, labels = ax.get_legend_handles_labels()
#     # Build (handle, numeric_value) pairs, ignoring labels with no trailing number
#     pairs = []
#     for handle, label in zip(handles, labels):
#         val = extract_chi_value(label)
#         if val is not None:
#             pairs.append((handle, label, val))
#         else:
#             # If no number is found, you can decide to put them at the end,
#             # or skip them, or handle differently:
#             pairs.append((handle, label, float('-inf')))

#     # Sort in descending order by the numeric value
#     pairs.sort(key=lambda x: x[2], reverse=True)

#     # Rebuild sorted handles and labels
#     sorted_handles = [p[0] for p in pairs]
#     sorted_labels = [p[1] for p in pairs]

#     ax.legend(sorted_handles, sorted_labels, fontsize='small', loc='best')

# #############################################
# # (a) and (b): 2x2 subplots for β = 1.0, split by lr.
# # Top row: Loss vs Epoch, bottom row: Fidelity vs Epoch.
# # Left column: lr = 0.01, Right column: lr = 0.001.
# #############################################

# fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex='col')
# for key, data in results.items():
#     chi, beta, lr = key
#     # Only consider trials with β = 1.0.
#     if beta == 1.0:
#         epochs = np.arange(len(data["loss_history"]))
#         # Choose subplot column based on lr.
#         if lr == 0.01:
#             ax_loss = axs[0, 0]
#             ax_fid = axs[1, 0]
#         elif lr == 0.001:
#             ax_loss = axs[0, 1]
#             ax_fid = axs[1, 1]
#         else:
#             continue  # skip other lr values if any
#         # Plot loss and fidelity for this chi.
#         ax_loss.plot(epochs, data["loss_history"],
#                      color=colors.get(chi, ['black'])[0],
#                      linestyle=colors.get(chi, [''])[1],
#                      label=f"$\\chi \\geq${chi}",
#                      linewidth=2.0)
#         ax_fid.plot(epochs, data["FA_history"],
#                     color=colors.get(chi, ['black'])[0],
#                     linestyle=colors.get(chi, [''])[1],
#                     label=f"$\\chi \\geq${chi}",
#                     linewidth=2.0)
# # For each axis, sort the legend and set larger font sizes.
# for ax in axs.flat:
#     sort_and_set_legend(ax)
#     ax.grid(alpha=0.15)
    
# # Set titles and labels with increased font sizes.
# axs[0, 0].set_title("Loss vs Epoch, $\\beta=1.0$, $\\eta=0.01$", fontsize=14)
# axs[0, 1].set_title("Loss vs Epoch, $\\beta=1.0$, $\\eta=0.001$", fontsize=14)
# axs[1, 0].set_title("Average Fidelity $\\langle F \\rangle$ vs Epoch, $\\beta=1.0$, $\\eta=0.01$", fontsize=14)
# axs[1, 1].set_title("Average Fidelity $\\langle F \\rangle$ vs Epoch, $\\beta=1.0$, $\\eta=0.001$", fontsize=14)
# for ax in axs[1, :]:
#     ax.set_xlabel("Epoch", fontsize=14)
# axs[0, 0].set_ylabel("Loss", fontsize=14)
# axs[1, 0].set_ylabel("$\\langle F \\rangle$", fontsize=14)
# plt.tight_layout()
# plt.savefig("../results/plots/loss_fidelity_lr.png")
# plt.show()

# #############################################
# # (c) and (d): 2x2 subplots for lr = 0.01, split by β.
# # Top row: Loss vs Epoch, bottom row: Fidelity vs Epoch.
# # Left column: β = 1.0, Right column: β = 8.0.
# #############################################

# fig2, axs2 = plt.subplots(2, 2, figsize=(12, 10), sharex='col')
# for key, data in results.items():
#     chi, beta, lr = key
#     # Only consider trials with lr = 0.01.
#     if lr == 0.01:
#         epochs = np.arange(len(data["loss_history"]))
#         if beta == 1.0:
#             ax_loss = axs2[0, 0]
#             ax_fid = axs2[1, 0]
#         elif beta == 8.0:
#             ax_loss = axs2[0, 1]
#             ax_fid = axs2[1, 1]
#         else:
#             continue
#         ax_loss.plot(epochs, data["loss_history"],
#                      color=colors.get(chi, ['black'])[0],
#                      linestyle=colors.get(chi, [''])[1],
#                      label=f"$\\chi \\geq${chi}",
#                      linewidth=2.0)
#         ax_fid.plot(epochs, data["FA_history"],
#                     color=colors.get(chi, ['black'])[0],
#                     linestyle=colors.get(chi, [''])[1],
#                     label=f"$\\chi \\geq${chi}",
#                     linewidth=2.0)
        
# # Sort legends for each subplot.
# for ax in axs2.flat:
#     sort_and_set_legend(ax)
#     ax.grid(alpha=0.15)

# axs2[0, 0].set_title("Loss vs Epoch, $\\beta=1.0$, $\\eta=0.01$", fontsize=14)
# axs2[0, 1].set_title("Loss vs Epoch, $\\beta=8.0$, $\\eta=0.01$", fontsize=14)
# axs2[1, 0].set_title("Average Fidelity $\\langle F \\rangle$ vs Epoch, $\\beta=1.0$, $\\eta=0.01$", fontsize=14)
# axs2[1, 1].set_title("Average Fidelity $\\langle F \\rangle$ vs Epoch, $\\beta=8.0$, $\\eta=0.01$", fontsize=14)
# for ax in axs2[1, :]:
#     ax.set_xlabel("Epoch", fontsize=14)
# axs2[0, 0].set_ylabel("Loss", fontsize=14)
# axs2[1, 0].set_ylabel("$\\langle F \\rangle$", fontsize=14)
# plt.tight_layout()
# plt.savefig("../results/plots/loss_fidelity_beta.png")
# plt.show()

# #############################################
# # (e): Bar chart of final fidelity vs $\chi$ for trials with lr=0.01 and β=8.0.
# # Also draw horizontal lines at $F = 5/6$ and $F = 2/3$.
# #############################################

# chi_vals = []
# final_fidelities = []
# for key, data in results.items():
#     chi, beta, lr = key
#     if lr == 0.01 and beta == 8.0:
#         chi_vals.append(chi)
#         final_fidelities.append(data["FA_history"][-1])  # last epoch fidelity

# # Sort by chi (if desired)
# chi_vals, final_fidelities = zip(*sorted(zip(chi_vals, final_fidelities)))
# colors_bar = {0.85: '#bc5090', 0.9: '#e5c185', 0.95: '#b8cdab', 0.99: '#74a892'}
# fig3, ax3 = plt.subplots(figsize=(8, 6))
# bars = ax3.bar([f"{c:.2f}" for c in chi_vals], final_fidelities,
#                color=[colors_bar.get(c, '#004343') for c in chi_vals])
# ax3.axhline(y=5/6, color='#c7522a', linestyle='--', label='$\\langle F \\rangle = \\frac{5}{6}$')
# ax3.axhline(y=3/4, color='#004343', linestyle='dashdot', label='$\\langle F \\rangle = \\frac{3}{4}$')
# ax3.set_xlabel("$\\chi$", fontsize=14)
# ax3.set_ylabel("$\\langle F \\rangle$", fontsize=14)
# ax3.set_title("Average Fidelity $\\langle F \\rangle$ vs $\\chi$, $\\beta=8.0$, $\\eta=0.01$", fontsize=14)
# ax3.legend(fontsize=10)
# ax3.set_ylim(0.6, 0.9)
# ax3.grid(alpha=0.15)
# plt.tight_layout()
# plt.savefig("../results/plots/final_fidelity_vs_chi_lr_001_beta_8.png")
# plt.show()

#############################################
# Plot mean loss and fidelity with error bars.
#############################################

# seed_ls = [1, 2, 3, 4]
# p_list = np.arange(0.01, 0.51, 0.01)

# for seed in seed_ls:
#     # Re-initialize lists for each seed
#     loss_ls = []
#     loss_mean_ls = []
#     loss_err_ls = []
#     fid_ls = []
#     fid_mean_ls = []
#     fid_err_ls = []

#     # Loop over p values
#     for count, p in enumerate(p_list):
#         if count < 9:
#             name = f"p_0_0{int(p * 100)}"
#         else:
#             name = f"p_0_{int(p * 100)}"
        
#         # Load results for this seed and p
#         filepath = f"../results/check_trial{seed}_noise/{name}_results.npy"
#         results = np.load(filepath, allow_pickle=True).item()
        
#         # Find the matching data by p-value
#         for key, data in results.items():
#             # Compare rounded values to avoid floating-point mismatch
#             if round(key[0], 2) == round(p, 2):
#                 l = data["loss_history"][-100:]
#                 f = data["FA_history"][-100:]
#                 print(f"Seed = {seed}, Found data for p = {p}: length of loss = {len(l)}")
                
#                 # Accumulate data
#                 loss_ls += l
#                 fid_ls += f
#                 # Compute means & standard errors for the last 100 points
#                 loss_mean_ls.append(np.mean(l))
#                 fid_mean_ls.append(np.mean(f))
#                 loss_err_ls.append(np.std(l) / np.sqrt(len(l)))
#                 fid_err_ls.append(np.std(f) / np.sqrt(len(f)))
#                 break

#     # Once we have all data for this seed, generate plots
#     fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

#     # Plot Fidelity
#     axs[0].scatter(p_list, fid_mean_ls, color='#c7522a', s=15, label="$\\langle F \\rangle$")
#     axs[0].plot(p_list, fid_mean_ls, color='#c7522a')
#     axs[0].set_title(f"Average Fidelity $\\langle F \\rangle$ vs $\\epsilon$ (seed={seed})", fontsize=16)
#     axs[0].set_ylabel("Average Fidelity $\\langle F \\rangle$", fontsize=14)
#     axs[0].legend(fontsize=14)
#     axs[0].grid(alpha=0.15)

#     # Plot Loss
#     axs[1].scatter(p_list, loss_mean_ls, color='#004343', s=15, marker='^', label="Loss")
#     axs[1].plot(p_list, loss_mean_ls, color='#004343')
#     axs[1].set_title("Average Loss vs $\\epsilon$", fontsize=14)
#     axs[1].set_ylabel("Average Loss", fontsize=14)
#     axs[1].set_xlabel("$\\epsilon$", fontsize=14)
#     axs[1].legend(fontsize=14)
#     axs[1].grid(alpha=0.15)

#     plt.tight_layout()
#     out_fig = f"../results/check_trial{seed}_noise/plot{seed}.png"
#     plt.savefig(out_fig)
#     plt.show()
#     plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

seed_ls = [1, 2, 3, 4]
p_list = np.arange(0.01, 0.51, 0.01)

# Define the three colours and markers to use for the three plots.
colors = ['#bc5090', '#e5c185', '#b8cdab']  # Total Loss, Cloning Loss, Average Fidelity
markers = ['^', 's', 'o']

for seed in seed_ls:
    # Initialize lists for each metric.
    loss_mean_ls = []
    loss_err_ls = []
    clone_loss_mean_ls = []
    clone_loss_err_ls = []
    FA_mean_ls = []
    FA_err_ls = []
    FB_mean_ls = []
    FB_err_ls = []
    
    # Loop over p values
    for count, p in enumerate(p_list):
        if count < 9:
            name = f"p_0_0{int(p * 100)}"
        else:
            name = f"p_0_{int(p * 100)}"
        
        # Load results for this seed and p
        filepath = f"../results/check_trial{seed}_noise/{name}_results.npy"
        results = np.load(filepath, allow_pickle=True).item()
        
        # Find the matching data by p-value
        for key, data in results.items():
            # Compare rounded values to avoid floating-point mismatch
            if round(key[0], 2) == round(p, 2):
                # Extract the last 100 values for each metric.
                l = data["loss_history"][-100:]
                c = data["cloning_loss_history"][-100:]
                fA = data["FA_history"][-100:]
                fB = data["FB_history"][-100:]
                print(f"Seed = {seed}, Found data for p = {p}: length of loss = {len(l)}")
                
                # Compute means & standard errors.
                loss_mean_ls.append(np.mean(l))
                loss_err_ls.append(np.std(l) / np.sqrt(len(l)))
                clone_loss_mean_ls.append(np.mean(c))
                clone_loss_err_ls.append(np.std(c) / np.sqrt(len(c)))
                FA_mean_ls.append(np.mean(fA))
                FA_err_ls.append(np.std(fA) / np.sqrt(len(fA)))
                FB_mean_ls.append(np.mean(fB))
                FB_err_ls.append(np.std(fB) / np.sqrt(len(fB)))
                break

    # Compute the overall (average) fidelity as the average of F_A and F_B.
    avg_fid_mean_ls = (np.array(FA_mean_ls) + np.array(FB_mean_ls)) / 2
    avg_fid_err_ls = (np.array(FA_err_ls) + np.array(FB_err_ls)) / 2

    # Create a figure with 3 rows (Total Loss, Cloning Loss, and Average Fidelity)
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Plot Total Loss.
    axs[0].scatter(p_list, loss_mean_ls, color=colors[0], s=15, marker=markers[0], label="Total Loss")
    axs[0].plot(p_list, loss_mean_ls, color=colors[0])
    axs[0].set_title(f"Total Loss vs $\\epsilon$ (seed={seed})", fontsize=16)
    axs[0].set_ylabel("Total Loss", fontsize=14)
    axs[0].legend(fontsize=14)
    axs[0].grid(alpha=0.15)
    
    # Plot Cloning Loss.
    axs[1].scatter(p_list, clone_loss_mean_ls, color=colors[1], s=15, marker=markers[1], label="Cloning Loss")
    axs[1].plot(p_list, clone_loss_mean_ls, color=colors[1])
    axs[1].set_title("Cloning Loss vs $\\epsilon$", fontsize=16)
    axs[1].set_ylabel("Cloning Loss", fontsize=14)
    axs[1].legend(fontsize=14)
    axs[1].grid(alpha=0.15)
    
    # Plot Average Fidelity.
    axs[2].scatter(p_list, avg_fid_mean_ls, color=colors[2], s=15, marker=markers[2], label="$\\langle F \\rangle$")
    axs[2].plot(p_list, avg_fid_mean_ls, color=colors[2])
    axs[2].set_title("Average Fidelity $\\langle F \\rangle$ vs $\\epsilon$", fontsize=16)
    axs[2].set_ylabel("$\\langle F \\rangle$", fontsize=14)
    axs[2].set_xlabel("$\\epsilon$", fontsize=14)
    axs[2].legend(fontsize=14)
    axs[2].grid(alpha=0.15)
    
    plt.tight_layout()
    out_fig = f"../results/check_trial{seed}_noise/plot{seed}.png"
    plt.savefig(out_fig)
    plt.show()
    plt.close(fig)