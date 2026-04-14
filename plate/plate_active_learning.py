"""
96-Well Plate Active Learning Pipeline
========================================
Integrates PlateSimulator → PlateDetector → BeliefModel (active learning).

The learner reads one well at a time (via the CV oracle) and updates the
shared BeliefModel from battleship_model.py, then uses its acquisition
function to pick the next most informative well to read.

BeliefModel is instantiated in *plate mode* (ship_sizes=[]):
  - Maintains Beta(α, β) posteriors per cell
  - Spatial Gaussian spreading shares evidence with neighbours
  - .prob_map, .get_entropy_map(), .select_query() are the same API
    used by the battleship experiments

Acquisition strategies (all from BeliefModel.select_query)
-----------------------------------------------------------
  random   : uniform random baseline
  grid     : systematic row-by-row scan
  prob     : argmax P(purple)  — exploitation
  entropy  : argmax H(p)       — uncertainty sampling
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from battleship_model import BeliefModel          # ← shared model
from config import STRATEGY_COLORS
from utils.plotting import interpolate_curve
from plate.plate_detector import PlateDetector
from plate.plate_simulator import PlateSimulator, clustered_plate_labels, random_plate_labels


# ── Strategy config ───────────────────────────────────────────────────────
STRATEGIES = ["random", "grid", "prob", "entropy"]

LABELS_STR = {
    "random":  "Random (baseline)",
    "grid":    "Systematic grid scan",
    "prob":    "Max Probability (exploit)",
    "entropy": "Max Entropy (uncertainty)",
}
COLORS = {s: STRATEGY_COLORS[s] for s in STRATEGIES}


# ══════════════════════════════════════════════════════════════════════════
# CV Oracle: simulator + detector
# ══════════════════════════════════════════════════════════════════════════

class PlateOracle:
    """
    Callable oracle that reads a well from the simulated plate image.

    oracle.query(row, col)  →  True (purple) / False (blue)

    Wraps PlateSimulator (image generation) + PlateDetector (CV pipeline).
    Falls back to ground truth when CV returns 'unknown'.
    """

    def __init__(self, ground_truth: np.ndarray, sim_seed: Optional[int] = None):
        self.ground_truth = ground_truth.astype(int)
        self.sim          = PlateSimulator(seed=sim_seed)
        self.image        = self.sim.generate_image(ground_truth)
        self.detector     = PlateDetector(geometry=self.sim.get_geometry())
        self.n_calls      = 0
        self.n_cv_errors  = 0

    def query(self, row: int, col: int) -> bool:
        self.n_calls += 1
        label, _, _ = self.detector.query_well(self.image, row, col)

        if label == "unknown":
            label = "purple" if self.ground_truth[row, col] == 1 else "blue"

        result = (label == "purple")
        if result != bool(self.ground_truth[row, col]):
            self.n_cv_errors += 1
        return result

    @property
    def cv_error_rate(self) -> float:
        return self.n_cv_errors / max(1, self.n_calls)


# ══════════════════════════════════════════════════════════════════════════
# Episode runner
# ══════════════════════════════════════════════════════════════════════════

def run_plate_episode(
    strategy:          str,
    ground_truth:      np.ndarray,
    seed:              int,
    prior_purple:      float = 0.25,
    spatial_sigma:     float = 1.5,
    use_cv_oracle:     bool  = True,
) -> Dict:
    """
    Run one full active-learning episode on a 96-well plate.

    Uses BeliefModel in plate mode (ship_sizes=[]) as the learner,
    and PlateOracle as the labelling oracle.

    Returns a dict with per-step history and summary statistics.
    """
    np.random.seed(seed)
    rows, cols = ground_truth.shape
    n_purple   = int(ground_truth.sum())

    # ── Shared BeliefModel (plate mode) ──────────────────────────────
    belief = BeliefModel(
        rows=rows, cols=cols,
        ship_sizes=[],            # activates plate mode
        prior_purple=prior_purple,
        spatial_sigma=spatial_sigma,
    )

    # ── Oracle ───────────────────────────────────────────────────────
    if use_cv_oracle:
        oracle   = PlateOracle(ground_truth, sim_seed=seed)
        query_fn = oracle.query
    else:
        oracle   = None
        query_fn = lambda r, c: bool(ground_truth[r, c])

    # Precompute grid scan order for 'grid' strategy
    grid_order = [(r, c) for r in range(rows) for c in range(cols)]

    history: List[Dict] = []
    step = 0

    while True:
        pos = belief.select_query(strategy, grid_order=grid_order)
        if pos is None:
            break
        row, col   = pos
        is_purple  = query_fn(row, col)

        # BeliefModel.update: is_hit = is_purple, no sunk_ship for plates
        belief.update(row, col, is_hit=is_purple)
        step += 1

        queried_mask   = np.zeros((rows, cols), dtype=bool)
        for r, c in belief._queried():
            queried_mask[r, c] = True

        purples_found = int((queried_mask & (ground_truth == 1)).sum())

        history.append({
            "step":           step,
            "row":            row,
            "col":            col,
            "is_purple":      is_purple,
            "purples_found":  purples_found,
            "frac_found":     purples_found / max(1, n_purple),
            "total_entropy":  float(belief.get_entropy_map().sum()),
            "max_prob":       float(belief.prob_map.max()),
        })

        if purples_found == n_purple:   # all positives found → early stop
            break

    cv_err = oracle.cv_error_rate if oracle is not None else 0.0

    return {
        "strategy":      strategy,
        "seed":          seed,
        "n_queries":     step,
        "n_purple":      n_purple,
        "purples_found": history[-1]["purples_found"] if history else 0,
        "cv_error_rate": cv_err,
        "history":       history,
        "belief":        belief,
        "oracle":        oracle,
    }


# ══════════════════════════════════════════════════════════════════════════
# Batch experiment
# ══════════════════════════════════════════════════════════════════════════

def run_plate_experiment(
    n_episodes:        int   = 100,
    strategies:        Optional[List[str]] = None,
    positive_fraction: float = 0.25,
    clustered:         bool  = True,
    use_cv_oracle:     bool  = True,
    verbose:           bool  = True,
) -> Dict[str, List[Dict]]:
    """
    Run n_episodes per strategy on randomly generated plates (same seeds).

    Parameters
    ----------
    clustered : if True, purple wells form spatial clusters (realistic);
                if False, wells are independently random.
    """
    if strategies is None:
        strategies = STRATEGIES

    results: Dict[str, List[Dict]] = {s: [] for s in strategies}

    for seed in range(n_episodes):
        gt = (
            clustered_plate_labels(positive_fraction=positive_fraction, seed=seed)
            if clustered
            else random_plate_labels(positive_fraction=positive_fraction, seed=seed)
        )

        for strategy in strategies:
            ep = run_plate_episode(strategy, gt, seed=seed,
                                   use_cv_oracle=use_cv_oracle)
            results[strategy].append(ep)

        if verbose and (seed + 1) % 25 == 0:
            print(f"\n[{seed + 1}/{n_episodes}]")
            for s in strategies:
                avg = np.mean([ep["n_queries"]     for ep in results[s]])
                err = np.mean([ep["cv_error_rate"] for ep in results[s]])
                print(f"  {LABELS_STR[s]:<30s}  avg_queries={avg:5.1f}  cv_err={err:.3f}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════

_interp_curve = interpolate_curve


def plot_plate_comparison(results: Dict[str, List[Dict]], save_dir: str = "."):
    strategies = list(results.keys())
    fig, axes  = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Plate Active Learning – Queries to Find All Purple Wells",
                 fontsize=13, fontweight="bold")

    data = [np.array([ep["n_queries"] for ep in results[s]]) for s in strategies]

    ax = axes[0]
    parts = ax.violinplot(data, showmedians=True)
    for pc, s in zip(parts["bodies"], strategies):
        pc.set_facecolor(COLORS[s]); pc.set_alpha(0.75)
    ax.set_xticks(range(1, len(strategies) + 1))
    ax.set_xticklabels([LABELS_STR[s] for s in strategies], rotation=14, ha="right")
    ax.set_ylabel("Queries"); ax.set_title("Distribution"); ax.grid(axis="y", alpha=0.3)
    for i, (s, d) in enumerate(zip(strategies, data)):
        ax.text(i + 1, d.max() + 0.5, f"μ={d.mean():.0f}",
                ha="center", fontsize=9, color=COLORS[s])

    ax = axes[1]
    for s, d in zip(strategies, data):
        q   = np.sort(d)
        cdf = np.arange(1, len(q) + 1) / len(q)
        ax.plot(q, cdf, label=LABELS_STR[s], color=COLORS[s], lw=2)
    ax.set_xlabel("Queries"); ax.set_ylabel("Cumulative probability")
    ax.set_title("CDF"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "plate_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()


def plot_plate_learning_curves(results: Dict[str, List[Dict]], save_dir: str = "."):
    strategies = list(results.keys())
    max_steps  = max(ep["n_queries"] for eps in results.values() for ep in eps)
    steps      = np.arange(1, max_steps + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Plate Active Learning Curves", fontsize=13, fontweight="bold")

    for s in strategies:
        frac_mat = np.array([_interp_curve(ep["history"], "frac_found",   max_steps)
                             for ep in results[s]])
        ent_mat  = np.array([_interp_curve(ep["history"], "total_entropy", max_steps)
                             for ep in results[s]])
        mu_f = frac_mat.mean(0); se_f = frac_mat.std(0) / np.sqrt(len(results[s]))
        mu_e = ent_mat.mean(0)

        axes[0].plot(steps, mu_f, label=LABELS_STR[s], color=COLORS[s], lw=2)
        axes[0].fill_between(steps, mu_f - se_f, mu_f + se_f,
                             color=COLORS[s], alpha=0.15)
        axes[1].plot(steps, mu_e, label=LABELS_STR[s], color=COLORS[s], lw=2)

    axes[0].set_xlabel("Queries"); axes[0].set_ylabel("Fraction of purple wells found")
    axes[0].set_title("Discovery rate"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Queries"); axes[1].set_ylabel("Total entropy (bits)")
    axes[1].set_title("Uncertainty reduction"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "plate_learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()


def plot_belief_evolution(episode: Dict, save_dir: str = "."):
    """Show prob_map (from BeliefModel) evolving alongside the plate image."""
    oracle   = episode.get("oracle")
    if oracle is None:
        return
    gt       = oracle.ground_truth
    image    = oracle.image
    sim      = oracle.sim
    strategy = episode["strategy"]
    n_steps  = episode["n_queries"]

    snap_at = sorted(set([
        0,
        max(0, n_steps // 4 - 1),
        max(0, n_steps // 2 - 1),
        max(0, 3 * n_steps // 4 - 1),
        n_steps - 1,
    ]))[:5]

    # Replay the episode to capture intermediate BeliefModel states
    np.random.seed(episode["seed"])
    belief2  = BeliefModel(rows=gt.shape[0], cols=gt.shape[1],
                           ship_sizes=[], prior_purple=0.25, spatial_sigma=1.5)
    oracle2  = PlateOracle(gt, sim_seed=episode["seed"])
    grid_ord = [(r, c) for r in range(gt.shape[0]) for c in range(gt.shape[1])]

    snapshots: List[Tuple] = []
    step = 0
    while True:
        pos = belief2.select_query(strategy, grid_order=grid_ord)
        if pos is None:
            break
        r, c   = pos
        is_p   = oracle2.query(r, c)
        belief2.update(r, c, is_hit=is_p)

        if step in snap_at:
            qmask = np.zeros(gt.shape, dtype=bool)
            for qr, qc in belief2._queried():
                qmask[qr, qc] = True
            snapshots.append((step + 1, belief2.prob_map.copy(), qmask.copy()))
        step += 1
        if step > snap_at[-1]:
            break

    import cv2
    rgb      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    row_lbl  = "ABCDEFGH"
    n_snaps  = len(snapshots)

    fig, axes = plt.subplots(2, n_snaps, figsize=(4 * n_snaps, 8))
    fig.suptitle(
        f"BeliefModel Evolution – strategy='{LABELS_STR[strategy]}', "
        f"seed={episode['seed']}, total={n_steps} queries",
        fontsize=11, fontweight="bold",
    )

    for ci, (step_n, prob, qmask) in enumerate(snapshots):
        ax_top = axes[0, ci]
        ax_top.imshow(rgb)
        ax_top.set_title(f"After {step_n} queries", fontsize=9)
        ax_top.axis("off")

        for ri in range(gt.shape[0]):
            for cj in range(gt.shape[1]):
                if qmask[ri, cj]:
                    cx  = sim.col_centers[cj]
                    cy  = sim.row_centers[ri]
                    col = "#00ff00" if gt[ri, cj] == 1 else "#ff4444"
                    ax_top.add_patch(plt.Circle(
                        (cx, cy), sim.well_radius * 0.8,
                        fill=False, edgecolor=col, linewidth=2
                    ))
        if ci == 0:
            ax_top.set_ylabel("Plate image", fontsize=9)

        ax_bot = axes[1, ci]
        ax_bot.imshow(prob, vmin=0, vmax=1, cmap="Purples", aspect="auto")
        ax_bot.contour(gt, levels=[0.5], colors="lime", linewidths=1.5, alpha=0.7)
        ax_bot.set_xticks(range(gt.shape[1]))
        ax_bot.set_yticks(range(gt.shape[0]))
        ax_bot.set_xticklabels(range(1, 13), fontsize=6)
        ax_bot.set_yticklabels(list(row_lbl), fontsize=6)
        if ci == 0:
            ax_bot.set_ylabel("BeliefModel P(purple)", fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, f"plate_belief_evolution_{strategy}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()


def print_summary(results: Dict[str, List[Dict]]):
    n_purple_avg = np.mean([ep["n_purple"] for ep in next(iter(results.values()))])
    print("\n" + "=" * 68)
    print(f"{'Strategy':<32} {'Mean':>6} {'Std':>6} {'Min':>5} {'Max':>5}  {'CV-Err':>7}")
    print("-" * 68)
    for s in results:
        q   = np.array([ep["n_queries"]     for ep in results[s]])
        err = np.array([ep["cv_error_rate"] for ep in results[s]])
        print(f"{LABELS_STR[s]:<32} {q.mean():>6.1f} {q.std():>6.1f} "
              f"{q.min():>5} {q.max():>5}  {err.mean():>7.3f}")
    print("=" * 68)
    print(f"\nPlate: 96 wells | ~{n_purple_avg:.0f} purple wells "
          f"({100*n_purple_avg/96:.0f}%) per plate\n")
