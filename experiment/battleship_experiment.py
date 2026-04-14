"""
Battleship Active Learning Experiment
======================================
Runs multiple episodes comparing acquisition strategies and visualises the results.

Metrics
-------
  n_queries          : total oracle calls to sink all ships  (lower = better)
  learning_curve     : fraction of ship cells found vs queries used
  entropy_curve      : total board entropy vs queries used
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from battleship_env import BattleshipBoard
from battleship_matrix_oracle import make_battleship_oracle
from battleship_model import Game
from config import STRATEGY_COLORS
from utils.plotting import interpolate_curve

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

STRATEGIES: List[str] = ["random", "prob", "entropy", "hunt_target", "pro_solver"]

LABELS: Dict[str, str] = {
    "random":      "Random (baseline)",
    "prob":        "Max Probability (exploit)",
    "entropy":     "Max Entropy (uncertainty)",
    "hunt_target": "Hunt-Target (heuristic)",
    "pro_solver": "Hunt/Target + Density (pro solver)",
}

COLORS: Dict[str, str] = {s: STRATEGY_COLORS[s] for s in STRATEGIES}


# ------------------------------------------------------------------
# Single episode
# ------------------------------------------------------------------

def run_episode(
    strategy: str,
    seed: int,
    board_rows: int = 8,
    board_cols: int = 10,
    oracle_mode: str = "board",
    *,
    rgb_l2_max: Optional[float] = None,
    rgb_per_channel_delta: Optional[float] = None,
) -> Dict:
    """
    Play one complete game with the given acquisition strategy.

    Returns a dict with per-step history and summary statistics.
    """
    board = BattleshipBoard(rows=board_rows, cols=board_cols, seed=seed)
    model = Game(board_rows=board_rows, board_cols=board_cols)
    oracle = make_battleship_oracle(
        board,
        seed=seed,
        oracle_mode=oracle_mode,
        rgb_l2_max=rgb_l2_max,
        rgb_per_channel_delta=rgb_per_channel_delta,
    )

    history: List[Dict] = []

    while not board.is_game_over():
        pos = model.select_query(strategy)
        if pos is None:
            break

        row, col = pos
        observed_hit, sunk_ship, actual_hit = oracle.query(row, col)
        model.update(row, col, observed_hit, sunk_ship)

        history.append({
            "step":            board.n_queries,
            "row":             row,
            "col":             col,
            "is_hit":          observed_hit,
            "observed_hit":    observed_hit,
            "actual_hit":      actual_hit,
            "ships_sunk":      len(board.get_sunk_ships()),
            "cells_found":     board.total_ship_cells - board.get_remaining_ship_cells(),
            "frac_found":      (board.total_ship_cells - board.get_remaining_ship_cells())
                               / board.total_ship_cells,
            "total_entropy":   model.get_entropy_map().sum(),
            "max_prob":        float(model.prob_map.max()),
        })

    return {
        "strategy":         strategy,
        "seed":             seed,
        "oracle_mode":      oracle_mode,
        "n_queries":        board.n_queries,
        "total_ship_cells": board.total_ship_cells,
        "cv_error_rate":    oracle.cv_error_rate,
        "unknown_rate":     oracle.unknown_rate,
        "history":          history,
        # keep final objects for detailed plots
        "board":            board,
        "model":            model,
    }


# ------------------------------------------------------------------
# Batch experiment
# ------------------------------------------------------------------

def run_experiment(
    n_episodes: int = 200,
    strategies: Optional[List[str]] = None,
    board_rows: int = 8,
    board_cols: int = 10,
    verbose: bool = True,
    oracle_mode: str = "board",
    *,
    rgb_l2_max: Optional[float] = None,
    rgb_per_channel_delta: Optional[float] = None,
) -> Dict[str, List[Dict]]:
    """
    Run *n_episodes* games per strategy, all on the same random seeds so
    board configurations are identical across strategies.
    """
    if strategies is None:
        strategies = STRATEGIES

    results: Dict[str, List[Dict]] = {s: [] for s in strategies}

    for seed in range(n_episodes):
        for strategy in strategies:
            ep = run_episode(
                strategy,
                seed=seed,
                board_rows=board_rows,
                board_cols=board_cols,
                oracle_mode=oracle_mode,
                rgb_l2_max=rgb_l2_max,
                rgb_per_channel_delta=rgb_per_channel_delta,
            )
            results[strategy].append(ep)

        if verbose and (seed + 1) % 50 == 0:
            print(f"\n[{seed + 1}/{n_episodes}]")
            for s in strategies:
                avg = np.mean([r["n_queries"] for r in results[s]])
                print(f"  {LABELS[s]:<35s}  avg queries = {avg:.1f}")

    return results


# ------------------------------------------------------------------
# Visualisation helpers
# ------------------------------------------------------------------

_interpolate_curve = interpolate_curve


def plot_comparison(results: Dict[str, List[Dict]], save_dir: str = "."):
    """
    Figure 1 – Distribution of queries to complete the game.
    Left : violin plot per strategy
    Right: cumulative distribution (CDF) of queries
    """
    strategies = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Active Learning Strategies – Queries to Win", fontsize=14, fontweight="bold")

    # ── Violin ───────────────────────────────────────────────────────
    ax = axes[0]
    data = [np.array([ep["n_queries"] for ep in results[s]]) for s in strategies]
    parts = ax.violinplot(data, positions=range(len(strategies)), showmedians=True)
    for pc, s in zip(parts["bodies"], strategies):
        pc.set_facecolor(COLORS[s])
        pc.set_alpha(0.75)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([LABELS[s] for s in strategies], rotation=12, ha="right")
    ax.set_ylabel("Number of queries")
    ax.set_title("Distribution")
    ax.grid(axis="y", alpha=0.3)

    # Add mean annotation
    for i, (s, d) in enumerate(zip(strategies, data)):
        ax.text(i, d.max() + 1, f"μ={d.mean():.0f}", ha="center", fontsize=9, color=COLORS[s])

    # ── CDF ──────────────────────────────────────────────────────────
    ax = axes[1]
    for s in strategies:
        q = np.sort([ep["n_queries"] for ep in results[s]])
        cdf = np.arange(1, len(q) + 1) / len(q)
        ax.plot(q, cdf, label=LABELS[s], color=COLORS[s], lw=2)
    ax.set_xlabel("Queries to sink all ships")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("CDF of queries to completion")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "battleship_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()


def plot_learning_curves(results: Dict[str, List[Dict]], save_dir: str = "."):
    """
    Figure 2 – Learning curves:
    Left : mean fraction of ship cells found vs queries
    Right: mean total board entropy vs queries
    """
    strategies = list(results.keys())
    max_steps = max(ep["n_queries"] for eps in results.values() for ep in eps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Active Learning Curves", fontsize=14, fontweight="bold")

    for s in strategies:
        frac_mat = np.array([
            _interpolate_curve(ep["history"], "frac_found", max_steps)
            for ep in results[s]
        ])
        ent_mat = np.array([
            _interpolate_curve(ep["history"], "total_entropy", max_steps)
            for ep in results[s]
        ])
        steps = np.arange(1, max_steps + 1)
        mean_frac = frac_mat.mean(axis=0)
        mean_ent  = ent_mat.mean(axis=0)
        se_frac   = frac_mat.std(axis=0) / np.sqrt(len(results[s]))

        axes[0].plot(steps, mean_frac, label=LABELS[s], color=COLORS[s], lw=2)
        axes[0].fill_between(steps,
                             mean_frac - se_frac,
                             mean_frac + se_frac,
                             color=COLORS[s], alpha=0.15)

        axes[1].plot(steps, mean_ent, label=LABELS[s], color=COLORS[s], lw=2)

    axes[0].set_xlabel("Number of queries (oracle calls)")
    axes[0].set_ylabel("Fraction of ship cells found")
    axes[0].set_title("Ship discovery rate")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Number of queries (oracle calls)")
    axes[1].set_ylabel("Total board entropy (bits)")
    axes[1].set_title("Uncertainty reduction over queries")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "battleship_learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()


def plot_episode_detail(episode: Dict, save_dir: str = "."):
    """
    Figure 3 – Snapshot grid for one episode.
    Shows board observation + probability map at 6 key steps.
    """
    board = episode["board"]
    nr, nc = board.rows, board.cols
    history = episode["history"]
    strategy = episode["strategy"]
    total_steps = len(history)

    # Replay the recorded observations so image-decoded runs stay faithful.
    replay_board = BattleshipBoard(rows=nr, cols=nc, seed=episode["seed"])
    replay_model = Game(board_rows=nr, board_cols=nc)
    replay_observed = np.full((nr, nc), -1, dtype=int)

    # Choose snapshot steps
    snap_steps = sorted(set([
        1,
        max(1, total_steps // 5),
        max(1, total_steps * 2 // 5),
        max(1, total_steps * 3 // 5),
        max(1, total_steps * 4 // 5),
        total_steps,
    ]))
    snap_steps = list(dict.fromkeys(snap_steps))[:6]  # deduplicate, max 6

    snapshots = []
    for entry in history:
        r, c = int(entry["row"]), int(entry["col"])
        observed_hit = bool(entry.get("observed_hit", entry["is_hit"]))
        actual_hit, actual_sunk = replay_board.query(r, c)
        sunk_for_model = actual_sunk if (observed_hit and actual_hit) else None
        replay_model.update(r, c, observed_hit, sunk_for_model)
        replay_observed[r, c] = int(observed_hit)
        step = int(entry["step"])
        if step in snap_steps:
            snapshots.append((
                step,
                replay_observed.copy(),
                replay_board.grid.copy(),
                replay_model.prob_map.copy(),
                replay_model.get_entropy_map().copy(),
            ))

    n_snaps = len(snapshots)
    fig, axes = plt.subplots(2, n_snaps, figsize=(3.5 * n_snaps, 7), squeeze=False)
    fig.suptitle(
        f"Episode Detail – strategy='{LABELS[strategy]}', seed={episode['seed']}, "
        f"total queries={total_steps}",
        fontsize=11, fontweight="bold",
    )

    obs_colors = plt.cm.get_cmap("RdYlGn", 3)  # miss=red, unknown=yellow, hit=green

    for col_i, (step_i, obs, grid, prob, ent) in enumerate(snapshots):
        # Top row: board state
        ax_top = axes[0, col_i]
        disp = obs.copy().astype(float)
        disp[obs == -1] = 0.5  # unknown → grey
        im = ax_top.imshow(disp, vmin=0, vmax=1, cmap="RdYlGn", interpolation="nearest")
        # Mark true ship positions (faint outline)
        for r in range(nr):
            for c in range(nc):
                if grid[r, c] == 1 and obs[r, c] == -1:
                    ax_top.add_patch(Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        linewidth=0.5, edgecolor="navy", facecolor="none", alpha=0.4
                    ))
        ax_top.set_title(f"Step {step_i}", fontsize=9)
        ax_top.set_xticks([]); ax_top.set_yticks([])
        if col_i == 0:
            ax_top.set_ylabel("Observed board", fontsize=9)

        # Bottom row: probability map
        ax_bot = axes[1, col_i]
        pm = ax_bot.imshow(prob, vmin=0, cmap="Blues", interpolation="nearest")
        ax_bot.set_xticks([]); ax_bot.set_yticks([])
        if col_i == 0:
            ax_bot.set_ylabel("P(ship) belief map", fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, f"battleship_episode_detail_{strategy}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()


# ------------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------------

def print_summary(results: Dict[str, List[Dict]]):
    print("\n" + "=" * 60)
    print(f"{'Strategy':<35} {'Mean':>6} {'Std':>6} {'Min':>5} {'Max':>5}")
    print("-" * 60)
    for s in results:
        q = np.array([ep["n_queries"] for ep in results[s]])
        print(f"{LABELS[s]:<35} {q.mean():>6.1f} {q.std():>6.1f} {q.min():>5} {q.max():>5}")
    print("=" * 60)
    ship_cells = results[list(results.keys())[0]][0]["total_ship_cells"]
    b0 = results[list(results.keys())[0]][0]["board"]
    board_cells = b0.rows * b0.cols
    print(
        f"\nBoard: {board_cells} cells ({b0.rows}×{b0.cols}) | Ship cells: {ship_cells} "
        f"({100*ship_cells/board_cells:.0f}%)\n"
    )
