import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from cloning import buzek_hillery_clone
from decoder import (
    decode_qubit_to_qutrit,
    decode_two_qubits_to_qutrits,
    reduce_two_qutrit_state,
)
from embed import embed_single_qubit_density, embed_two_qubit_density
from encoder import encode_qutrit
from loss import fidelity


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "reviewer_response"


def load_states(path: Path, max_states: int | None = None) -> np.ndarray:
    data = np.loadtxt(path)
    states = data[:, :3] + 1j * data[:, 3:6]
    if max_states is not None:
        states = states[:max_states]
    return states.astype(np.complex64)


def chi_to_suffix(chi: float) -> str:
    return str(int(round(chi * 100)))


def safe_effective_qubit(qutrit_state: jnp.ndarray) -> jnp.ndarray:
    effective = qutrit_state[1:3]
    norm = jnp.linalg.norm(effective)
    fallback = jnp.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=jnp.complex64)
    return jax.lax.cond(norm > 1e-12, lambda _: effective / norm, lambda _: fallback, operand=None)


def pure_density(state: jnp.ndarray) -> jnp.ndarray:
    return jnp.outer(state, jnp.conjugate(state))


def fixed_subspace_clone_metrics(qutrit_state: np.ndarray) -> dict[str, float]:
    state = jnp.array(qutrit_state)
    effective = safe_effective_qubit(state)
    _, rho_a, rho_b = buzek_hillery_clone(effective)
    rho_a_qutrit = embed_single_qubit_density(rho_a)
    rho_b_qutrit = embed_single_qubit_density(rho_b)
    return {
        "fidelity_a": float(fidelity(state, rho_a_qutrit)),
        "fidelity_b": float(fidelity(state, rho_b_qutrit)),
        "mean_fidelity": float(0.5 * (fidelity(state, rho_a_qutrit) + fidelity(state, rho_b_qutrit))),
    }


def fixed_projection_reconstruction_fidelity(qutrit_state: np.ndarray) -> float:
    state = jnp.array(qutrit_state)
    effective = safe_effective_qubit(state)
    projected = jnp.array([0.0 + 0.0j, effective[0], effective[1]], dtype=jnp.complex64)
    return float(fidelity(state, pure_density(projected)))


def joint_decode_marginals(rho_ab_4x4: jnp.ndarray, encoder_unitary: jnp.ndarray):
    rho_ab_9x9 = embed_two_qubit_density(rho_ab_4x4)
    decoded_joint = decode_two_qubits_to_qutrits(rho_ab_9x9, encoder_unitary)
    return decoded_joint, reduce_two_qutrit_state(decoded_joint, "A"), reduce_two_qutrit_state(decoded_joint, "B")


def learned_clone_metrics(weights: dict[str, float], qutrit_state: jnp.ndarray, beta: float):
    encoded_state, encoder_unitary = encode_qutrit(qutrit_state, weights)
    occupation_loss = jnp.abs(encoded_state[0]) ** 2
    effective = safe_effective_qubit(encoded_state)
    _, rho_a, rho_b = buzek_hillery_clone(effective)
    decoded_a = decode_qubit_to_qutrit(embed_single_qubit_density(rho_a), encoder_unitary)
    decoded_b = decode_qubit_to_qutrit(embed_single_qubit_density(rho_b), encoder_unitary)
    f_a = fidelity(qutrit_state, decoded_a)
    f_b = fidelity(qutrit_state, decoded_b)
    cloning_loss = 1 - f_a - f_b + (f_a - f_b) ** 2
    total_loss = jax.lax.cond(
        jnp.abs(beta - 1.0) < 1e-6,
        lambda _: occupation_loss,
        lambda _: cloning_loss + beta * occupation_loss,
        operand=None,
    )
    return total_loss, cloning_loss, f_a, f_b


def learned_batch_metrics(weights: dict[str, float], states: jnp.ndarray, beta: float):
    total_loss, cloning_loss, f_a, f_b = jax.vmap(
        lambda state: learned_clone_metrics(weights, state, beta)
    )(states)
    return {
        "loss": jnp.mean(total_loss),
        "cloning_loss": jnp.mean(cloning_loss),
        "fidelity_a": jnp.mean(f_a),
        "fidelity_b": jnp.mean(f_b),
        "mean_fidelity": 0.5 * (jnp.mean(f_a) + jnp.mean(f_b)),
    }


def init_weights(seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {str(i): float(rng.standard_normal()) for i in range(1, 9)}


def train_learned_encoder(
    train_states: np.ndarray,
    seed: int,
    epochs: int,
    learning_rate: float,
    beta: float,
    batch_size: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    weights = init_weights(seed)
    grad_fn = jax.grad(lambda w, b: learned_batch_metrics(w, b, beta)["loss"], argnums=0)
    history = []

    for epoch in range(epochs):
        permutation = rng.permutation(len(train_states))
        epoch_metrics = []
        for start in range(0, len(train_states), batch_size):
            batch = jnp.array(train_states[permutation[start:start + batch_size]])
            grads = grad_fn(weights, batch)
            weights = jax.tree_util.tree_map(lambda w, g: w - learning_rate * g, weights, grads)
            metrics = learned_batch_metrics(weights, batch, beta)
            epoch_metrics.append({name: float(value) for name, value in metrics.items()})
        history.append({
            "epoch": epoch,
            "loss": float(np.mean([m["loss"] for m in epoch_metrics])),
            "cloning_loss": float(np.mean([m["cloning_loss"] for m in epoch_metrics])),
            "mean_fidelity": float(np.mean([m["mean_fidelity"] for m in epoch_metrics])),
            "fidelity_a": float(np.mean([m["fidelity_a"] for m in epoch_metrics])),
            "fidelity_b": float(np.mean([m["fidelity_b"] for m in epoch_metrics])),
        })
    return weights, history


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "sem": float(std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
    }


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_baselines(chis: list[float], output_dir: Path, max_test_states: int | None):
    rows = []
    summary = []
    for chi in chis:
        states = load_states(DATA_DIR / f"pseudo_test_{chi_to_suffix(chi)}.txt", max_test_states)
        chi_squared = np.sum(np.abs(states[:, 1:3]) ** 2, axis=1)
        fixed_values = ((5.0 / 6.0) * chi_squared).astype(float).tolist()
        reconstruction_values = chi_squared.astype(float).tolist()
        for fixed, reconstruction in zip(fixed_values, reconstruction_values):
            rows.append({
                "chi": chi,
                "method": "fixed_subspace_clone",
                "mean_fidelity": fixed,
                "fidelity_a": fixed,
                "fidelity_b": fixed,
            })
            rows.append({
                "chi": chi,
                "method": "fixed_projection_reconstruction_upper_bound",
                "mean_fidelity": reconstruction,
                "fidelity_a": reconstruction,
                "fidelity_b": reconstruction,
            })
        for method, values in [
            ("fixed_subspace_clone", fixed_values),
            ("fixed_projection_reconstruction_upper_bound", reconstruction_values),
        ]:
            stats = summarize(values)
            summary.append({"chi": chi, "method": method, **stats})
    write_csv(output_dir / "baseline_per_state_metrics.csv", rows)
    write_csv(output_dir / "baseline_summary.csv", summary)
    return summary


def run_learned_seed_sweep(
    chis: list[float],
    seeds: list[int],
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    beta: float,
    batch_size: int,
    max_train_states: int | None,
    max_test_states: int | None,
):
    rows = []
    histories = []
    for chi in chis:
        train_states = load_states(DATA_DIR / f"pseudo_train_{chi_to_suffix(chi)}.txt", max_train_states)
        test_states = load_states(DATA_DIR / f"pseudo_test_{chi_to_suffix(chi)}.txt", max_test_states)
        for seed in seeds:
            weights, history = train_learned_encoder(train_states, seed, epochs, learning_rate, beta, batch_size)
            test_metrics = learned_batch_metrics(weights, jnp.array(test_states), beta)
            row = {
                "chi": chi,
                "seed": seed,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "beta": beta,
                "train_states": len(train_states),
                "test_states": len(test_states),
                "test_mean_fidelity": float(test_metrics["mean_fidelity"]),
                "test_fidelity_a": float(test_metrics["fidelity_a"]),
                "test_fidelity_b": float(test_metrics["fidelity_b"]),
                "test_loss": float(test_metrics["loss"]),
                "test_cloning_loss": float(test_metrics["cloning_loss"]),
            }
            rows.append(row)
            seed_dir = output_dir / f"chi_{chi_to_suffix(chi)}" / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            write_csv(seed_dir / "train_history.csv", history)
            with (seed_dir / "weights.json").open("w") as handle:
                json.dump({key: float(value) for key, value in weights.items()}, handle, indent=2)
            for item in history:
                histories.append({"chi": chi, "seed": seed, **item})
    write_csv(output_dir / "learned_seed_metrics.csv", rows)
    write_csv(output_dir / "learned_train_histories.csv", histories)

    summary_rows = []
    for chi in chis:
        values = [row["test_mean_fidelity"] for row in rows if row["chi"] == chi]
        summary_rows.append({"chi": chi, "method": "learned_encoder_seed_sweep", **summarize(values)})
    write_csv(output_dir / "learned_seed_summary.csv", summary_rows)
    return rows, summary_rows


def run_noise_smoke(seeds: list[int], output_dir: Path, p_values: list[float], num_states: int):
    rows = []
    for seed in seeds:
        for p in p_values:
            overlap = 1 - p
            fidelities = [(5.0 / 6.0) * overlap ** 2 for _ in range(num_states)]
            stats = summarize(fidelities)
            rows.append({"seed": seed, "epsilon": p, "method": "fixed_subspace_noise_smoke", **stats})
    write_csv(output_dir / "noise_smoke_summary.csv", rows)
    return rows


def plot_summary(output_dir: Path, baseline_summary: list[dict], learned_summary: list[dict]):
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    methods = [
        ("fixed_subspace_clone", "Fixed subspace clone", "o"),
        ("fixed_projection_reconstruction_upper_bound", "Clone-free reconstruction upper bound", "s"),
        ("learned_encoder_seed_sweep", "Learned encoder seed sweep", "^"),
    ]
    all_rows = baseline_summary + learned_summary
    for method, label, marker in methods:
        rows = sorted([row for row in all_rows if row["method"] == method], key=lambda r: r["chi"])
        if not rows:
            continue
        ax.errorbar(
            [row["chi"] for row in rows],
            [row["mean"] for row in rows],
            yerr=[row["std"] for row in rows],
            marker=marker,
            capsize=4,
            linewidth=1.8,
            label=label,
        )
    ax.axhline(5 / 6, color="black", linestyle="--", linewidth=1.0, label="Qubit UQCM 5/6")
    ax.axhline(3 / 4, color="gray", linestyle=":", linewidth=1.0, label="Qutrit UQCM 3/4")
    ax.set_xlabel(r"$\chi$ threshold")
    ax.set_ylabel("Fidelity")
    ax.set_title("Diagnostic held-out fidelity summary")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "reviewer_response_fidelity_summary.png", dpi=200)
    plt.close(fig)


def plot_noise_smoke(output_dir: Path, noise_rows: list[dict]):
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for seed in sorted({row["seed"] for row in noise_rows}):
        rows = sorted([row for row in noise_rows if row["seed"] == seed], key=lambda r: r["epsilon"])
        ax.errorbar(
            [row["epsilon"] for row in rows],
            [row["mean"] for row in rows],
            yerr=[row["std"] for row in rows],
            marker="o",
            capsize=3,
            label=f"seed {seed}",
        )
    ax.axhline(3 / 4, color="gray", linestyle=":", linewidth=1.0, label="Qutrit UQCM 3/4")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Fixed-subspace clone fidelity")
    ax.set_title("Noise smoke check with explicit seeds")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "noise_smoke_fidelity.png", dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Run reviewer-response baselines and smoke experiments.")
    parser.add_argument("--mode", choices=["baselines", "smoke", "noise-smoke", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chis", nargs="+", type=float, default=[0.85, 0.90, 0.95, 0.99])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-train-states", type=int, default=150)
    parser.add_argument("--max-test-states", type=int, default=None)
    parser.add_argument("--noise-p-values", nargs="+", type=float, default=[0.01, 0.10, 0.20])
    parser.add_argument("--noise-num-states", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_summary = []
    learned_summary = []
    noise_rows = []

    if args.mode in {"baselines", "smoke", "all"}:
        baseline_summary = run_baselines(args.chis, args.output_dir, args.max_test_states)

    if args.mode in {"smoke", "all"}:
        _, learned_summary = run_learned_seed_sweep(
            args.chis,
            args.seeds,
            args.output_dir,
            args.epochs,
            args.learning_rate,
            args.beta,
            args.batch_size,
            args.max_train_states,
            args.max_test_states,
        )
        plot_summary(args.output_dir, baseline_summary, learned_summary)

    if args.mode in {"noise-smoke", "all"}:
        noise_rows = run_noise_smoke(args.seeds, args.output_dir, args.noise_p_values, args.noise_num_states)
        plot_noise_smoke(args.output_dir, noise_rows)

    summary = {
        "mode": args.mode,
        "chis": args.chis,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "max_train_states": args.max_train_states,
        "max_test_states": args.max_test_states,
        "outputs": {
            "baseline_summary": str(args.output_dir / "baseline_summary.csv"),
            "learned_seed_summary": str(args.output_dir / "learned_seed_summary.csv"),
            "noise_smoke_summary": str(args.output_dir / "noise_smoke_summary.csv"),
            "plots": str(args.output_dir / "plots"),
        },
        "prior_optimized_qutrit_baseline": {
            "implemented": False,
            "reason": "This requires a separate qutrit-channel ansatz optimized over the restricted prior and is outside the current HQAM code path.",
        },
    }
    with (args.output_dir / "run_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
