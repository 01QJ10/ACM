import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("../results")
PLOTS_DIR = RESULTS_DIR / "plots"
TRIAL1_RESULTS_PATH = RESULTS_DIR / "trial1" / "hyperparams_results.npy"
TRIAL2_NOISE_DIR = RESULTS_DIR / "trial2_noise"
MEAN_LOSS_FID_RESULTS_PATH = RESULTS_DIR / "check_trial5_noise" / "hyperparams_results.npy"
REPORT_SUFFIX = "_large_text"

CHI_STYLES = {
    0.99: ("#b8cdab", "solid"),
    0.95: ("#004343", "dotted"),
    0.90: ("#e5c185", "dashed"),
    0.85: ("#c7522a", "dashdot"),
}
BAR_COLORS = {0.85: "#bc5090", 0.90: "#e5c185", 0.95: "#b8cdab", 0.99: "#74a892"}

TITLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 16
LEGEND_FONT_SIZE = 14
LINE_WIDTH = 2.2
MARKER_SIZE = 6


def load_results(path: Path):
    return np.load(path, allow_pickle=True).item()


def extract_chi_value(label: str):
    match = re.search(r"([\d.]+)$", label)
    return float(match.group(1)) if match else None


def sort_and_set_legend(ax, loc: str = "best"):
    handles, labels = ax.get_legend_handles_labels()
    pairs = []
    for handle, label in zip(handles, labels):
        chi_value = extract_chi_value(label)
        sort_key = chi_value if chi_value is not None else float("-inf")
        pairs.append((sort_key, handle, label))

    pairs.sort(key=lambda item: item[0], reverse=True)
    sorted_handles = [item[1] for item in pairs]
    sorted_labels = [item[2] for item in pairs]
    ax.legend(sorted_handles, sorted_labels, fontsize=LEGEND_FONT_SIZE, loc=loc)


def style_axis(ax):
    ax.grid(alpha=0.15)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)


def save_figure(fig, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.0, h_pad=2.0, w_pad=1.0)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_report_figures(
    results_path: Path = TRIAL1_RESULTS_PATH,
    output_dir: Path = PLOTS_DIR,
    suffix: str = REPORT_SUFFIX,
):
    results = load_results(results_path)
    plot_loss_fidelity_by_lr(results, output_dir / f"loss_fidelity_lr{suffix}.png")
    plot_loss_fidelity_by_beta(results, output_dir / f"loss_fidelity_beta{suffix}.png")
    plot_final_fidelity_vs_chi(
        results,
        output_dir / f"final_fidelity_vs_chi_lr_001_beta_8{suffix}.png",
    )


def plot_loss_fidelity_by_lr(results, output_path: Path):
    fig, axs = plt.subplots(2, 2, figsize=(14, 12), sharex="col")

    for (chi, beta, lr), data in results.items():
        if beta != 1.0:
            continue

        epochs = np.arange(len(data["loss_history"]))
        if lr == 0.01:
            ax_loss = axs[0, 0]
            ax_fid = axs[1, 0]
        elif lr == 0.001:
            ax_loss = axs[0, 1]
            ax_fid = axs[1, 1]
        else:
            continue

        color, linestyle = CHI_STYLES.get(chi, ("black", "solid"))
        ax_loss.plot(
            epochs,
            data["loss_history"],
            color=color,
            linestyle=linestyle,
            label=f"$\\chi \\geq${chi}",
            linewidth=LINE_WIDTH,
        )
        ax_fid.plot(
            epochs,
            data["FA_history"],
            color=color,
            linestyle=linestyle,
            label=f"$\\chi \\geq${chi}",
            linewidth=LINE_WIDTH,
        )

    for ax in axs.flat:
        sort_and_set_legend(ax)
        style_axis(ax)

    axs[0, 0].set_title("Loss vs Epoch, $\\beta=1.0$, $\\eta=0.01$", fontsize=TITLE_FONT_SIZE)
    axs[0, 1].set_title("Loss vs Epoch, $\\beta=1.0$, $\\eta=0.001$", fontsize=TITLE_FONT_SIZE)
    axs[1, 0].set_title(
        "Average Fidelity $\\langle F \\rangle$ vs Epoch, $\\beta=1.0$, $\\eta=0.01$",
        fontsize=TITLE_FONT_SIZE,
    )
    axs[1, 1].set_title(
        "Average Fidelity $\\langle F \\rangle$ vs Epoch, $\\beta=1.0$, $\\eta=0.001$",
        fontsize=TITLE_FONT_SIZE,
    )

    for ax in axs[1, :]:
        ax.set_xlabel("Epoch", fontsize=LABEL_FONT_SIZE)
    axs[0, 0].set_ylabel("Loss", fontsize=LABEL_FONT_SIZE)
    axs[1, 0].set_ylabel("$\\langle F \\rangle$", fontsize=LABEL_FONT_SIZE)

    save_figure(fig, output_path)


def plot_loss_fidelity_by_beta(results, output_path: Path):
    fig, axs = plt.subplots(2, 2, figsize=(14, 12), sharex="col")

    for (chi, beta, lr), data in results.items():
        if lr != 0.01:
            continue

        epochs = np.arange(len(data["loss_history"]))
        if beta == 1.0:
            ax_loss = axs[0, 0]
            ax_fid = axs[1, 0]
        elif beta == 8.0:
            ax_loss = axs[0, 1]
            ax_fid = axs[1, 1]
        else:
            continue

        color, linestyle = CHI_STYLES.get(chi, ("black", "solid"))
        ax_loss.plot(
            epochs,
            data["loss_history"],
            color=color,
            linestyle=linestyle,
            label=f"$\\chi \\geq${chi}",
            linewidth=LINE_WIDTH,
        )
        ax_fid.plot(
            epochs,
            data["FA_history"],
            color=color,
            linestyle=linestyle,
            label=f"$\\chi \\geq${chi}",
            linewidth=LINE_WIDTH,
        )

    for ax in axs.flat:
        sort_and_set_legend(ax)
        style_axis(ax)

    axs[0, 0].set_title("Loss vs Epoch, $\\beta=1.0$, $\\eta=0.01$", fontsize=TITLE_FONT_SIZE)
    axs[0, 1].set_title("Loss vs Epoch, $\\beta=8.0$, $\\eta=0.01$", fontsize=TITLE_FONT_SIZE)
    axs[1, 0].set_title(
        "Average Fidelity $\\langle F \\rangle$ vs Epoch, $\\beta=1.0$, $\\eta=0.01$",
        fontsize=TITLE_FONT_SIZE,
    )
    axs[1, 1].set_title(
        "Average Fidelity $\\langle F \\rangle$ vs Epoch, $\\beta=8.0$, $\\eta=0.01$",
        fontsize=TITLE_FONT_SIZE,
    )

    for ax in axs[1, :]:
        ax.set_xlabel("Epoch", fontsize=LABEL_FONT_SIZE)
    axs[0, 0].set_ylabel("Loss", fontsize=LABEL_FONT_SIZE)
    axs[1, 0].set_ylabel("$\\langle F \\rangle$", fontsize=LABEL_FONT_SIZE)

    save_figure(fig, output_path)


def plot_final_fidelity_vs_chi(results, output_path: Path):
    chi_vals = []
    final_fidelities = []
    for (chi, beta, lr), data in results.items():
        if lr == 0.01 and beta == 8.0:
            chi_vals.append(chi)
            final_fidelities.append(data["FA_history"][-1])

    sorted_pairs = sorted(zip(chi_vals, final_fidelities))
    chi_vals = [pair[0] for pair in sorted_pairs]
    final_fidelities = [pair[1] for pair in sorted_pairs]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(
        [f"{chi:.2f}" for chi in chi_vals],
        final_fidelities,
        color=[BAR_COLORS.get(chi, "#004343") for chi in chi_vals],
    )
    ax.axhline(
        y=5 / 6,
        color="#c7522a",
        linestyle="--",
        label="$\\langle F \\rangle = \\frac{5}{6}$",
    )
    ax.axhline(
        y=3 / 4,
        color="#004343",
        linestyle="dashdot",
        label="$\\langle F \\rangle = \\frac{3}{4}$",
    )
    ax.set_xlabel("$\\chi$", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("$\\langle F \\rangle$", fontsize=LABEL_FONT_SIZE)
    ax.set_title(
        "Average Fidelity $\\langle F \\rangle$ vs $\\chi$, $\\beta=8.0$, $\\eta=0.01$",
        fontsize=TITLE_FONT_SIZE,
    )
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    ax.set_ylim(0.6, 0.9)
    style_axis(ax)

    save_figure(fig, output_path)


def plot_mean_loss_fid_vs_p(
    results_path: Path = MEAN_LOSS_FID_RESULTS_PATH,
    output_path: Path = PLOTS_DIR / f"mean_loss_fid_vs_p{REPORT_SUFFIX}.png",
):
    results = load_results(results_path)
    sorted_results = sorted(
        [
            (key, data)
            for key, data in results.items()
            if key[0] <= 0.20 and key[1] == 8.0 and key[2] == 0.01
        ],
        key=lambda item: item[0][0],
    )
    p_list = [item[0][0] for item in sorted_results]
    fid_mean_ls = []
    loss_mean_ls = []

    for _, data in sorted_results:
        fid_history = data["FA_history"][-100:]
        loss_history = data["loss_history"][-100:]
        fid_mean_ls.append(np.mean(fid_history))
        loss_mean_ls.append(np.mean(loss_history))

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(
        p_list,
        fid_mean_ls,
        "o-",
        color="#c7522a",
        markersize=4,
        linewidth=LINE_WIDTH,
        label="$\\langle F \\rangle$",
    )
    axs[0].set_title(
        "Average Fidelity $\\langle F \\rangle$ vs $p$",
        fontsize=TITLE_FONT_SIZE,
    )
    axs[0].set_ylabel("Average Fidelity $\\langle F \\rangle$", fontsize=LABEL_FONT_SIZE)
    axs[0].legend(fontsize=LEGEND_FONT_SIZE)
    style_axis(axs[0])

    axs[1].plot(
        p_list,
        loss_mean_ls,
        "^-",
        color="#004343",
        markersize=4,
        linewidth=LINE_WIDTH,
        label="Loss",
    )
    axs[1].set_title("Average Loss vs $p$", fontsize=TITLE_FONT_SIZE)
    axs[1].set_ylabel("Average Loss", fontsize=LABEL_FONT_SIZE)
    axs[1].set_xlabel("$p$", fontsize=LABEL_FONT_SIZE)
    axs[1].legend(fontsize=LEGEND_FONT_SIZE)
    style_axis(axs[1])

    save_figure(fig, output_path)


def plot_seed_noise_metrics(seed_ls=(5, 6)):
    uqcm_qubit = 5 / 6
    uqcm_qutrit = 0.75

    for seed in seed_ls:
        results_path = RESULTS_DIR / f"check_trial{seed}_noise" / "hyperparams_results.npy"
        results = load_results(results_path)

        filtered_keys = [key for key in results if key[1] == 8.0 and key[2] == 0.01]
        filtered_keys = sorted(filtered_keys, key=lambda key: key[0])
        p_list = [key[0] for key in filtered_keys]

        total_fid_mean_ls = []
        f_cl_rho_ls = []
        f_cl_rho0_ls = []

        for key in filtered_keys:
            data = results[key]
            fa_history = data["FA_history"]
            last_fa = fa_history[-100:] if len(fa_history) >= 100 else fa_history
            total_fid_mean_ls.append(np.mean(last_fa))
            f_cl_rho_ls.append(data["F_cl_rho"])
            f_cl_rho0_ls.append(data["F_cl_rho0"])

        colors = ["#bc5090", "#e5c185", "#b8cdab"]
        markers = ["^", "s", "o"]

        fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        axs[0].plot(
            p_list,
            total_fid_mean_ls,
            f"{markers[0]}-",
            color=colors[0],
            markersize=MARKER_SIZE,
            label="$F_{overall}$",
        )
        axs[0].axhline(uqcm_qubit, color="k", linestyle="-.", label="UQCM Qubit")
        axs[0].axhline(uqcm_qutrit, color="gray", linestyle="--", label="UQCM Qutrit")
        axs[0].set_title(f"$F_{{overall}}$ vs $\\epsilon$ (seed={seed})", fontsize=TITLE_FONT_SIZE)
        axs[0].set_ylabel("$F_{overall}$", fontsize=LABEL_FONT_SIZE)
        axs[0].legend(fontsize=LEGEND_FONT_SIZE)
        style_axis(axs[0])

        axs[1].plot(
            p_list,
            f_cl_rho_ls,
            f"{markers[1]}-",
            color=colors[1],
            markersize=MARKER_SIZE,
            label="$F_{qubit}$",
        )
        axs[1].axhline(uqcm_qubit, color="k", linestyle="-.", label="UQCM Qubit")
        axs[1].axhline(uqcm_qutrit, color="gray", linestyle="--", label="UQCM Qutrit")
        axs[1].set_title("$F_{qubit}$ vs $\\epsilon$", fontsize=TITLE_FONT_SIZE)
        axs[1].set_ylabel("$F_{qubit}$", fontsize=LABEL_FONT_SIZE)
        axs[1].legend(fontsize=LEGEND_FONT_SIZE)
        style_axis(axs[1])

        axs[2].plot(
            p_list,
            f_cl_rho0_ls,
            f"{markers[2]}-",
            color=colors[2],
            markersize=MARKER_SIZE,
            label="$F_{qutrit}$",
        )
        axs[2].axhline(uqcm_qubit, color="k", linestyle="-.", label="UQCM Qubit")
        axs[2].axhline(uqcm_qutrit, color="gray", linestyle="--", label="UQCM Qutrit")
        axs[2].set_title("$F_{qutrit}$ vs $\\epsilon$", fontsize=TITLE_FONT_SIZE)
        axs[2].set_ylabel("$F_{qutrit}$", fontsize=LABEL_FONT_SIZE)
        axs[2].set_xlabel("$\\epsilon$", fontsize=LABEL_FONT_SIZE)
        axs[2].legend(fontsize=LEGEND_FONT_SIZE)
        style_axis(axs[2])

        output_path = RESULTS_DIR / f"check_trial{seed}_noise" / f"plot{seed}.png"
        save_figure(fig, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ACM plotting artifacts.")
    parser.add_argument(
        "--report-large-text",
        action="store_true",
        help="Generate larger-font copies of the three report figures in results/plots.",
    )
    parser.add_argument(
        "--mean-noise-large-text",
        action="store_true",
        help="Generate a larger-font copy of mean_loss_fid_vs_p in results/plots.",
    )
    parser.add_argument(
        "--seed-noise",
        action="store_true",
        help="Generate the seed 5 and 6 epsilon sweep plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not any([args.report_large_text, args.mean_noise_large_text, args.seed_noise]):
        args.seed_noise = True

    if args.report_large_text:
        plot_report_figures()
    if args.mean_noise_large_text:
        plot_mean_loss_fid_vs_p()
    if args.seed_noise:
        plot_seed_noise_metrics()


if __name__ == "__main__":
    main()
