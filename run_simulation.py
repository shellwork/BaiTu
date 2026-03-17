"""
run_simulation.py — Closed-loop campaign runner for BaiTu.

Executes the active learning loop against the EnzymeKineticsSimulator and
produces five diagnostic figures in the plots/ directory, plus a JSON
history file consumed by qc_dashboard.py.

Usage
-----
  python run_simulation.py [--cycles 5] [--query_size 32] [--ensemble 5]
                           [--epochs 30] [--seed_size 50] [--val_size 100]
                           [--output_dir plots] [--pt_path data/kinetics_simulated_with_embeddings.pt]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as tud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import Config
from simulator import EnzymeKineticsSimulator
from qc_framework import QCMonitor, THRESHOLDS
from active_learning import (
    score_pool_contribution,
    select_top_k,
    ensemble_predict,
    ContributionWeights,
)
from train import train_deep_ensemble, evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loader(dataset, batch_size: int = 32, shuffle: bool = False):
    return tud.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _log_scale_mae(model, dataloader, device: str) -> Tuple[float, float]:
    """
    Return MAE of log_kcat and log_km predictions (natural log units).
    Unlike evaluate(), this stays in log space — matching the stopping criterion.
    """
    model.eval()
    log_kcat_targets, log_kcat_preds = [], []
    log_km_targets,   log_km_preds   = [], []

    with torch.no_grad():
        for batch in dataloader:
            enzyme_embed   = batch["enzyme_embed"].to(device)
            substrate_fp   = batch["substrate_fp"].to(device)
            substrate_conc = batch["substrate_conc"].to(device)
            enzyme_conc    = batch["enzyme_conc"].to(device)
            mask = batch["has_param_label"].squeeze(-1) > 0
            if not mask.any():
                continue
            out = model(enzyme_embed, substrate_fp, substrate_conc=substrate_conc, enzyme_conc=enzyme_conc)
            log_kcat_targets.append(batch["log_kcat"][mask].cpu().numpy().flatten())
            log_kcat_preds.append(out["log_kcat"][mask].cpu().numpy().flatten())
            log_km_targets.append(batch["log_km"][mask].cpu().numpy().flatten())
            log_km_preds.append(out["log_km"][mask].cpu().numpy().flatten())

    if not log_kcat_targets:
        return float("nan"), float("nan")

    t_kcat = np.concatenate(log_kcat_targets)
    p_kcat = np.concatenate(log_kcat_preds)
    t_km   = np.concatenate(log_km_targets)
    p_km   = np.concatenate(log_km_preds)

    mae_kcat = float(np.mean(np.abs(t_kcat - p_kcat)))
    mae_km   = float(np.mean(np.abs(t_km   - p_km)))
    return mae_kcat, mae_km


def _mean_pool_variance(ensemble, pool_loader, device: str) -> float:
    """Mean ensemble variance over v0 for all pool samples."""
    vars_: List[float] = []
    for batch in pool_loader:
        pred = ensemble_predict(
            ensemble=ensemble,
            enzyme_embed=batch["enzyme_embed"].to(device),
            substrate_fp=batch["substrate_fp"].to(device),
            substrate_conc=batch["substrate_conc"].to(device),
            enzyme_conc=batch["enzyme_conc"].to(device),
        )
        vars_.extend(pred["v0_pred_var"].cpu().numpy().flatten().tolist())
    return float(np.mean(vars_)) if vars_ else 0.0


def _pool_predictions(ensemble, pool_loader, device: str) -> np.ndarray:
    """Collect mean v0_pred for all pool samples (for stability tracking)."""
    preds: List[float] = []
    for batch in pool_loader:
        pred = ensemble_predict(
            ensemble=ensemble,
            enzyme_embed=batch["enzyme_embed"].to(device),
            substrate_fp=batch["substrate_fp"].to(device),
            substrate_conc=batch["substrate_conc"].to(device),
            enzyme_conc=batch["enzyme_conc"].to(device),
        )
        preds.extend(pred["v0_pred_mean"].cpu().numpy().flatten().tolist())
    return np.array(preds)


def _prediction_stability(prev: Optional[np.ndarray], curr: np.ndarray) -> float:
    """Fraction of pool predictions that changed by < 1 % relative to previous cycle."""
    if prev is None or len(prev) == 0 or len(curr) == 0:
        return 0.0
    # Align lengths (pool shrinks each cycle as queried samples are removed)
    n = min(len(prev), len(curr))
    p, c = prev[:n], curr[:n]
    denom = np.abs(p) + 1e-12
    rel_change = np.abs(c - p) / denom
    return float(np.mean(rel_change < 0.01))


def _information_gain(var_prev: float, var_curr: float) -> float:
    """Entropy reduction: ΔH = 0.5 * log(var_prev / var_curr) (nats)."""
    if var_prev <= 0 or var_curr <= 0:
        return 0.0
    return max(0.0, 0.5 * math.log(var_prev / var_curr))


def _collect_kcat_km_preds(ensemble, pool_loader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Collect mean kcat / km predictions for biological bounds QC check."""
    kcats, kms = [], []
    for batch in pool_loader:
        pred = ensemble_predict(
            ensemble=ensemble,
            enzyme_embed=batch["enzyme_embed"].to(device),
            substrate_fp=batch["substrate_fp"].to(device),
        )
        kcats.extend(pred["kcat_mean"].cpu().numpy().flatten().tolist())
        kms.extend(pred["km_mean"].cpu().numpy().flatten().tolist())
    return np.array(kcats), np.array(kms)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PALETTE = {
    "blue":   "#2563EB",
    "orange": "#EA580C",
    "green":  "#16A34A",
    "red":    "#DC2626",
    "purple": "#7C3AED",
    "grey":   "#6B7280",
}


def plot_learning_curves(history: List[Dict], output_dir: str) -> str:
    """Figure 1: R² and linear MAE over AL cycles (3-panel)."""
    cycles = [h["cycle"] + 1 for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Learning Curves over Active Learning Cycles", fontsize=13, y=1.02)

    metrics_r2  = [("v0_r2",   "v₀ (rate)",   PALETTE["blue"]),
                   ("kcat_r2", "k_cat",        PALETTE["orange"]),
                   ("km_r2",   "K_m",          PALETTE["green"])]
    metrics_mae = [("v0_mae",   "v₀ (rate)",   PALETTE["blue"]),
                   ("kcat_mae", "k_cat",        PALETTE["orange"]),
                   ("km_mae",   "K_m",          PALETTE["green"])]

    # Panel 1: R² for all three targets
    ax = axes[0]
    for key, label, color in metrics_r2:
        vals = [h.get(key, float("nan")) for h in history]
        ax.plot(cycles, vals, marker="o", color=color, label=label, linewidth=2)
    ax.axhline(0, color=PALETTE["grey"], linestyle="--", linewidth=0.8)
    ax.set_xlabel("Active Learning Cycle")
    ax.set_ylabel("R²")
    ax.set_title("Held-out R² Score")
    ax.set_ylim(-0.2, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: MAE (log-scale for kcat/km)
    ax = axes[1]
    for key, label, color in [("mae_log_kcat", "log k_cat", PALETTE["orange"]),
                                ("mae_log_km",   "log K_m",  PALETTE["green"])]:
        vals = [h.get(key, float("nan")) for h in history]
        ax.plot(cycles, vals, marker="s", color=color, label=label, linewidth=2)
    ax.axhline(THRESHOLDS["mae_log_kcat_threshold"], color=PALETTE["red"],
               linestyle="--", linewidth=1.5, label="Stop threshold")
    ax.set_xlabel("Active Learning Cycle")
    ax.set_ylabel("MAE (log units)")
    ax.set_title("Log-scale MAE (k_cat & K_m)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Training set size
    ax = axes[2]
    n_labeled = [h.get("n_labeled", 0) for h in history]
    ax.bar(cycles, n_labeled, color=PALETTE["blue"], alpha=0.7, edgecolor="white")
    ax.set_xlabel("Active Learning Cycle")
    ax.set_ylabel("Labeled Examples")
    ax.set_title("Cumulative Labeled Training Data")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(output_dir, "learning_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved {out}")
    return out


def plot_stopping_criteria(history: List[Dict], output_dir: str) -> str:
    """Figure 2: Stopping-criterion metrics with threshold lines (3-panel)."""
    cycles = [h["cycle"] + 1 for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Stopping-Criterion Metrics over Active Learning Cycles", fontsize=13, y=1.02)

    # Panel 1: Held-out log MAE (kcat)
    ax = axes[0]
    mae_kcat = [h.get("mae_log_kcat", float("nan")) for h in history]
    ax.plot(cycles, mae_kcat, marker="o", color=PALETTE["orange"], linewidth=2, label="log k_cat MAE")
    ax.axhline(THRESHOLDS["mae_log_kcat_threshold"], color=PALETTE["red"],
               linestyle="--", linewidth=2, label=f"Stop threshold ({THRESHOLDS['mae_log_kcat_threshold']} nats)")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("MAE (log units, natural)")
    ax.set_title("Criterion 2: Held-Out Prediction Error")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Prediction stability
    ax = axes[1]
    stability = [h.get("prediction_stability_pct", 0.0) * 100 for h in history]
    ax.plot(cycles, stability, marker="s", color=PALETTE["purple"], linewidth=2, label="Stability %")
    ax.axhline(THRESHOLDS["prediction_stability"] * 100, color=PALETTE["red"],
               linestyle="--", linewidth=2,
               label=f"Stop threshold ({THRESHOLDS['prediction_stability']:.0%})")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Prediction Stability (%)")
    ax.set_title("Criterion 3: Prediction Stability")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Cycle count budget
    ax = axes[2]
    ax.bar(cycles, [1] * len(cycles), color=PALETTE["blue"], alpha=0.6, width=0.6, label="Completed cycle")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlim(0.5, THRESHOLDS["max_cycles"] + 0.5)
    ax.set_ylim(0, 1.5)
    ax.set_xticks(range(1, int(THRESHOLDS["max_cycles"]) + 1))
    ax.set_xlabel("Cycle")
    ax.set_ylabel("")
    ax.set_title(f"Criterion 1: Budget (max {int(THRESHOLDS['max_cycles'])} cycles)")
    # Mark budget limit
    ax.axvline(THRESHOLDS["max_cycles"] + 0.5, color=PALETTE["red"],
               linestyle="--", linewidth=2, label="Budget limit")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    # Annotate stopping point if one was triggered
    for h in history:
        if h.get("stop_triggered", False):
            for ax in axes[:2]:
                ax.axvline(h["cycle"] + 1, color=PALETTE["green"], linestyle=":",
                           linewidth=2, alpha=0.8)

    plt.tight_layout()
    out = os.path.join(output_dir, "stopping_criteria.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved {out}")
    return out


def plot_uncertainty_reduction(history: List[Dict], output_dir: str) -> str:
    """Figure 3: Mean ensemble variance over pool per cycle with exponential fit."""
    cycles = np.array([h["cycle"] + 1 for h in history], dtype=float)
    var_means = np.array([h.get("mean_ensemble_var", float("nan")) for h in history])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cycles, var_means, marker="o", color=PALETTE["blue"], linewidth=2,
            markersize=8, label="Mean pool variance")

    # Exponential decay fit (if enough points and no NaN)
    valid = ~np.isnan(var_means)
    if valid.sum() >= 3 and np.all(var_means[valid] > 0):
        try:
            log_var = np.log(var_means[valid])
            coeffs = np.polyfit(cycles[valid], log_var, 1)
            x_fit = np.linspace(cycles[valid].min(), cycles[valid].max(), 100)
            y_fit = np.exp(np.polyval(coeffs, x_fit))
            ax.plot(x_fit, y_fit, "--", color=PALETTE["orange"], linewidth=1.5,
                    label=f"Exp. fit  τ={-1/coeffs[0]:.1f} cycles")
        except Exception:
            pass

    ax.set_xlabel("Active Learning Cycle")
    ax.set_ylabel("Mean Ensemble Variance (v₀)")
    ax.set_title("Uncertainty Reduction over Active Learning Cycles")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    out = os.path.join(output_dir, "uncertainty_reduction.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved {out}")
    return out


def plot_information_gain(history: List[Dict], output_dir: str) -> str:
    """Figure 4: Information gain (entropy reduction) per cycle — bar chart."""
    cycles = [h["cycle"] + 1 for h in history]
    gains  = [h.get("info_gain_nats", 0.0) for h in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [PALETTE["green"] if g > 0.01 else PALETTE["grey"] for g in gains]
    bars = ax.bar(cycles, gains, color=colors, edgecolor="white", width=0.6)

    # Add value labels on top of bars
    for bar, g in zip(bars, gains):
        if g > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{g:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Active Learning Cycle")
    ax.set_ylabel("Information Gain ΔH (nats)")
    ax.set_title("Information Gain per Cycle\n(Entropy Reduction in Ensemble Uncertainty)")
    ax.set_xticks(cycles)
    ax.grid(True, alpha=0.3, axis="y")

    green_patch = mpatches.Patch(color=PALETTE["green"], label="Positive gain")
    grey_patch  = mpatches.Patch(color=PALETTE["grey"],  label="Negligible gain")
    ax.legend(handles=[green_patch, grey_patch], fontsize=9)

    plt.tight_layout()
    out = os.path.join(output_dir, "information_gain.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved {out}")
    return out


def plot_qc_timeline(history: List[Dict], output_dir: str) -> str:
    """Figure 5: QC metric heatmap over cycles (rows = metrics, cols = cycles)."""
    qc_keys = [
        ("qc_replicate_cv_ok",      "Replicate CV < 10%"),
        ("qc_kcat_bounds_ok",       "k_cat in bounds"),
        ("qc_km_bounds_ok",         "K_m in bounds"),
        ("qc_ensemble_var_ok",      "Ensemble var > 0"),
        ("qc_query_size_ok",        "Query size valid"),
        ("qc_no_hardware_errors",   "No hardware errors"),
    ]

    n_cycles = len(history)
    n_metrics = len(qc_keys)
    matrix = np.zeros((n_metrics, n_cycles))

    for j, h in enumerate(history):
        for i, (key, _) in enumerate(qc_keys):
            matrix[i, j] = 1.0 if h.get(key, True) else 0.0

    fig, ax = plt.subplots(figsize=(max(6, n_cycles * 1.2), max(3, n_metrics * 0.7)))
    cmap = matplotlib.colors.ListedColormap([PALETTE["red"], PALETTE["green"]])
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_cycles))
    ax.set_xticklabels([f"Cycle {h['cycle']+1}" for h in history])
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels([label for _, label in qc_keys])
    ax.set_title("QC Metrics Timeline (Green = Pass, Red = Fail)")

    # Add text annotations
    for i in range(n_metrics):
        for j in range(n_cycles):
            ax.text(j, i, "✓" if matrix[i, j] else "✗",
                    ha="center", va="center", fontsize=14,
                    color="white", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(output_dir, "qc_timeline.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved {out}")
    return out


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation(
    max_cycles: int = 5,
    query_size: int = 32,
    n_ensemble: int = 5,
    epochs_per_cycle: int = 30,
    seed_size: int = 50,
    val_size: int = 100,
    output_dir: str = "plots",
    pt_path: str = Config.PREPROCESSED_DATA_PATH,
    device: str = Config.DEVICE,
) -> List[Dict]:
    """
    Run the closed-loop active learning simulation.

    Returns
    -------
    history : list of per-cycle metric dicts
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Initialise simulator ────────────────────────────────────────────
    sim = EnzymeKineticsSimulator(
        pt_path=pt_path,
        seed_size=seed_size,
        val_size=val_size,
    )

    qc_monitor = QCMonitor()
    history: List[Dict] = []

    # Starting labeled set (seed) and fixed held-out validation
    labeled_dataset = sim.get_seed_dataset()
    val_dataset     = sim.get_held_out_dataset()
    val_loader      = _make_loader(val_dataset, batch_size=32)

    prev_pool_preds: Optional[np.ndarray] = None
    prev_mean_var: float = float("nan")

    print(f"\n{'='*60}")
    print(f"  BaiTu Closed-Loop Simulation")
    print(f"  Cycles: {max_cycles}  |  Query/cycle: {query_size}  |  Ensemble: {n_ensemble}")
    print(f"  Seed: {seed_size}  |  Val: {val_size}  |  Device: {device}")
    print(f"{'='*60}\n")

    for cycle in range(max_cycles):
        print(f"\n{'─'*50}")
        print(f"  Cycle {cycle + 1}/{max_cycles}  |  Labeled: {len(labeled_dataset)}")
        print(f"{'─'*50}")

        # ── 2. Train deep ensemble ─────────────────────────────────────────
        ensemble = train_deep_ensemble(
            train_dataset=labeled_dataset,
            val_dataset=val_dataset,
            n_members=n_ensemble,
            batch_size=32,
            lr=Config.LEARNING_RATE,
            num_epochs=epochs_per_cycle,
            device=device,
        )

        # ── 3. Evaluate on held-out set ────────────────────────────────────
        val_metrics = evaluate(ensemble[0], val_loader, device)
        mae_log_kcat, mae_log_km = _log_scale_mae(ensemble[0], val_loader, device)

        print(f"  Val R²(v0)={val_metrics.get('v0_r2', float('nan')):.4f} "
              f"| MAE log(kcat)={mae_log_kcat:.4f} | MAE log(km)={mae_log_km:.4f}")

        # ── 4. Score remaining pool ────────────────────────────────────────
        pool_dataset   = sim.get_pool_dataset()
        pool_positions = sim.get_pool_positions_remaining()

        if len(pool_positions) < query_size:
            print(f"  Pool exhausted ({len(pool_positions)} < {query_size}). Stopping.")
            break

        pool_loader    = _make_loader(pool_dataset, batch_size=32)
        labeled_loader = _make_loader(labeled_dataset, batch_size=32)

        contrib = score_pool_contribution(
            model=ensemble[0],
            ensemble=ensemble,
            pool_loader=pool_loader,
            labeled_loader=labeled_loader,
            device=device,
            weights=ContributionWeights(),
        )
        top_k_pool_positions_idx = select_top_k(contrib, k=query_size)
        # top_k_pool_positions_idx are positions within the *remaining* pool array
        # we need to map back to actual pool_positions
        selected_pool_positions = [pool_positions[i] for i in top_k_pool_positions_idx]

        # ── 5. QC checks ──────────────────────────────────────────────────
        replicate_cv = sim.get_replicate_cv(selected_pool_positions, n_replicates=3)

        # Get model predictions on the pool for biological bounds check
        kcat_preds, km_preds = _collect_kcat_km_preds(ensemble, pool_loader, device)
        mean_var = _mean_pool_variance(ensemble, pool_loader, device)

        qc_report = qc_monitor.run_cycle_checks(
            cycle=cycle,
            replicate_cv=replicate_cv,
            kcat_preds=kcat_preds,
            km_preds=km_preds,
            mean_ensemble_var=mean_var,
            query_size=query_size,
            hardware_errors=0,  # simulated: no hardware errors
        )

        if qc_report.flags:
            print(f"  QC flags: {qc_report.flags}")

        # ── 6. Compute prediction stability ───────────────────────────────
        curr_pool_preds = _pool_predictions(ensemble, pool_loader, device)
        stability = _prediction_stability(prev_pool_preds, curr_pool_preds)
        prev_pool_preds = curr_pool_preds

        # ── 7. Information gain ───────────────────────────────────────────
        info_gain = _information_gain(prev_mean_var, mean_var)
        prev_mean_var = mean_var

        # ── 8. Oracle labeling ────────────────────────────────────────────
        new_data = sim.query(selected_pool_positions)
        labeled_dataset = tud.ConcatDataset([labeled_dataset, new_data])

        # ── 9. Record history ─────────────────────────────────────────────
        record: Dict = {
            "cycle": cycle,
            "n_labeled": len(labeled_dataset),
            # Held-out metrics (linear scale)
            **{k: float(v) for k, v in val_metrics.items()},
            # Log-scale MAE (for stopping criterion)
            "mae_log_kcat": float(mae_log_kcat),
            "mae_log_km":   float(mae_log_km),
            # Uncertainty / info gain
            "mean_ensemble_var":        float(mean_var),
            "info_gain_nats":           float(info_gain),
            "prediction_stability_pct": float(stability),
            # QC flags
            "qc_replicate_cv_ok":    bool(qc_report.replicate_cv_ok),
            "qc_kcat_bounds_ok":     bool(qc_report.kcat_bounds_ok),
            "qc_km_bounds_ok":       bool(qc_report.km_bounds_ok),
            "qc_ensemble_var_ok":    bool(qc_report.ensemble_var_gt_zero),
            "qc_query_size_ok":      bool(qc_report.query_size_ok),
            "qc_no_hardware_errors": bool(qc_report.hardware_errors == 0),
            "qc_all_ok":             bool(qc_report.all_ok),
            "qc_flags":              qc_report.flags,
            "stop_triggered":        False,
        }

        # ── 10. Check stopping criteria ───────────────────────────────────
        history.append(record)
        decision = qc_monitor.should_stop(history)
        if decision.should_stop:
            record["stop_triggered"] = True
            print(f"\n  STOPPING CRITERION TRIGGERED: {decision.reason}")
            print(f"  Metrics at stop: {decision.metrics_at_stop}")
            break

    print(f"\n{'='*60}")
    print(f"  Simulation complete. Total cycles: {len(history)}")
    print(f"  Final labeled size: {len(labeled_dataset)}")
    print(f"{'='*60}\n")

    # ── Save history to JSON ───────────────────────────────────────────────
    history_path = os.path.join(output_dir, "simulation_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Simulation] History saved to {history_path}")

    # ── Save QC plan markdown ──────────────────────────────────────────────
    plan_md = qc_monitor.generate_markdown_report(history)
    plan_path = os.path.join(output_dir, "qc_stopping_criteria_plan.md")
    with open(plan_path, "w") as f:
        f.write(plan_md)
    print(f"[Simulation] QC plan saved to {plan_path}")

    # ── Generate all plots ─────────────────────────────────────────────────
    plot_learning_curves(history, output_dir)
    plot_stopping_criteria(history, output_dir)
    plot_uncertainty_reduction(history, output_dir)
    plot_information_gain(history, output_dir)
    plot_qc_timeline(history, output_dir)

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BaiTu Closed-Loop Campaign Simulator")
    parser.add_argument("--cycles",      type=int,   default=5,    help="Maximum AL cycles")
    parser.add_argument("--query_size",  type=int,   default=32,   help="Samples queried per cycle")
    parser.add_argument("--ensemble",    type=int,   default=5,    help="Ensemble members")
    parser.add_argument("--epochs",      type=int,   default=30,   help="Training epochs per cycle")
    parser.add_argument("--seed_size",   type=int,   default=50,   help="Initial seed set size")
    parser.add_argument("--val_size",    type=int,   default=100,  help="Held-out validation size")
    parser.add_argument("--output_dir",  type=str,   default="plots", help="Output directory")
    parser.add_argument("--pt_path",     type=str,   default=Config.PREPROCESSED_DATA_PATH)
    args = parser.parse_args()

    run_simulation(
        max_cycles=args.cycles,
        query_size=args.query_size,
        n_ensemble=args.ensemble,
        epochs_per_cycle=args.epochs,
        seed_size=args.seed_size,
        val_size=args.val_size,
        output_dir=args.output_dir,
        pt_path=args.pt_path,
    )
