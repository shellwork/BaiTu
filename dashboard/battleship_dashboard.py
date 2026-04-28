"""
BaiTu Dashboard — multi-page Streamlit app.

Run (from repository root):
    streamlit run dashboard/battleship_dashboard.py

Pages (sidebar)
---------------
- **Home**          – project overview and pointers.
- **Simulator**     – four acquisition strategies on an in-browser
                      Battleship simulation; hunt-target view on a plate +
                      hits-vs-round chart; unclear CV readings (score
                      0.4 – 0.6) pause for a human hit / miss call.
- **OT-2 Hardware** – launch, pause, resume, stop, and reset the real
                      ``hardware.battleship_ot2_loop`` subprocess. Live
                      plate state, belief map, latest camera frame, step
                      history, and log stream.
"""

from __future__ import annotations

import os
import datetime
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.battleship_env import BattleshipBoard
from core.battleship_model import Game
from config import DEFAULT_SHIP_SIZES, BOARD_ROWS, BOARD_COLS, PLATE_ROWS, PLATE_COLS, STRATEGY_COLORS
from experiment.battleship_synthetic import (
    NoiseConfig,
    generate_single_well_reading,
)
from dashboard.ot2_controller import LoopConfig, OT2Controller
from dashboard.live_camera import LivePreview
from hardware.battleship_ot2_loop import DEFAULT_DECK

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

STRATEGIES = ["random", "prob", "entropy", "hunt_target"]
LABELS = {
    "random":      "Random",
    "prob":        "Max Probability",
    "entropy":     "Max Entropy",
    "hunt_target": "Hunt-Target",
}
COLORS = {s: STRATEGY_COLORS[s] for s in STRATEGIES}

SHIP_SIZES = DEFAULT_SHIP_SIZES
SHIP_NAMES = ["Carrier", "Battleship", "Cruiser", "Submarine", "Destroyer"]

# Liquid parameters
LIQUID_START_ML = 15  # Starting cabbage juice in mL
LIQUID_PER_SHOT_UL = 100  # Liquid consumed per shot in µL

RESULTS_DIR = Path("battleship_results")
RESULTS_DIR.mkdir(exist_ok=True)

ROW_LABELS = list("ABCDEFGH")  # standard 96-well row labels


# ------------------------------------------------------------------
# Helpers – initialise session state per tab
# ------------------------------------------------------------------

def _make_fresh_state(seed: int, prefix: str = "") -> dict:
    """Create a dict of fresh game objects for all 4 strategies."""
    boards, models = {}, {}
    for s in STRATEGIES:
        boards[s] = BattleshipBoard(rows=BOARD_ROWS, cols=BOARD_COLS, seed=seed)
        models[s] = Game(board_rows=BOARD_ROWS, board_cols=BOARD_COLS)

    noise_cfg = NoiseConfig(seed=seed + 1000, hit_std=0.20, miss_std=0.20)  # Higher noise for more unclear readings
    rng = np.random.default_rng(seed + 2000)

    # Liquid tracking: convert mL to µL for tracking
    LIQUID_START_UL = LIQUID_START_ML * 1000

    return {
        f"{prefix}boards": boards,
        f"{prefix}models": models,
        f"{prefix}round": 0,
        f"{prefix}hits_per_round": {s: [] for s in STRATEGIES},
        f"{prefix}unclear_count_per_round": {s: [] for s in STRATEGIES},
        f"{prefix}cv_scores": {s: np.full((BOARD_ROWS, BOARD_COLS), np.nan) for s in STRATEGIES},
        f"{prefix}binary_calls": {s: np.full((BOARD_ROWS, BOARD_COLS), -2, dtype=int) for s in STRATEGIES},
        f"{prefix}shot_order": {s: [] for s in STRATEGIES},
        f"{prefix}cv_readings": {s: [] for s in STRATEGIES},  # track all CV readings for variance
        f"{prefix}previously_unclear_cells": {s: set() for s in STRATEGIES},  # track cells that were unclear
        f"{prefix}liquid_remaining_per_round": {s: [LIQUID_START_UL] for s in STRATEGIES},  # in µL
        f"{prefix}noise_cfg": noise_cfg,
        f"{prefix}rng": rng,
        f"{prefix}seed": seed,
        f"{prefix}next_seed": seed + 1,
        f"{prefix}done": False,
        f"{prefix}display_strategy": "hunt_target",
        f"{prefix}unclear_queue": [],  # for failure-handling tab
        f"{prefix}paused_for_user": False,
        f"{prefix}history_log": [],   # for saving
        f"{prefix}auto_play": False,  # for slow auto-play mode
        f"{prefix}resume_auto_play": False,  # resume after unclear dialog
    }


def _init_session(prefix: str = ""):
    """Populate st.session_state with defaults if needed."""
    if f"{prefix}boards" not in st.session_state:
        seed = st.session_state.get(f"{prefix}seed", 42)
        for k, v in _make_fresh_state(seed, prefix).items():
            st.session_state[k] = v
    # Ensure next_seed is always set
    if f"{prefix}next_seed" not in st.session_state:
        st.session_state[f"{prefix}next_seed"] = st.session_state.get(f"{prefix}seed", 42) + 1
    # Ensure auto_play is always set
    if f"{prefix}auto_play" not in st.session_state:
        st.session_state[f"{prefix}auto_play"] = False
    # Ensure resume_auto_play is always set
    if f"{prefix}resume_auto_play" not in st.session_state:
        st.session_state[f"{prefix}resume_auto_play"] = False


def _reset_session(seed: int, prefix: str = ""):
    """Hard-reset all game state with the given seed."""
    for k, v in _make_fresh_state(seed, prefix).items():
        st.session_state[k] = v
    # Increment the next seed for convenience
    st.session_state[f"{prefix}next_seed"] = seed + 1


# ------------------------------------------------------------------
# Step logic: advance one round across all strategies (no QC check)
# ------------------------------------------------------------------

def _step_all_strategies(prefix: str = ""):
    """Fire one shot for each strategy; record CV reading & hits."""
    boards = st.session_state[f"{prefix}boards"]
    models = st.session_state[f"{prefix}models"]
    cfg = st.session_state[f"{prefix}noise_cfg"]
    rng = st.session_state[f"{prefix}rng"]

    any_active = False
    round_num = st.session_state[f"{prefix}round"] + 1

    for s in STRATEGIES:
        board = boards[s]
        model = models[s]

        if board.is_game_over():
            # carry forward last count
            prev = st.session_state[f"{prefix}hits_per_round"][s]
            last_hits = prev[-1] if prev else 0
            st.session_state[f"{prefix}hits_per_round"][s].append(last_hits)
            # carry forward unclear count
            prev_unclear = st.session_state[f"{prefix}unclear_count_per_round"][s]
            last_unclear = prev_unclear[-1] if prev_unclear else 0
            st.session_state[f"{prefix}unclear_count_per_round"][s].append(last_unclear)
            # carry forward liquid
            prev_liquid = st.session_state[f"{prefix}liquid_remaining_per_round"][s]
            last_liquid = prev_liquid[-1] if prev_liquid else 0
            st.session_state[f"{prefix}liquid_remaining_per_round"][s].append(last_liquid)
            continue

        any_active = True
        pos = model.select_query(s)
        if pos is None:
            continue
        row, col = pos

        is_hit_true = bool(board.grid[row, col])

        # generate synthetic CV reading
        score, unclear, binary_call = generate_single_well_reading(is_hit_true, cfg, rng)

        # Track cumulative unclear count
        prev_unclear = st.session_state[f"{prefix}unclear_count_per_round"][s]
        cumulative_unclear = (prev_unclear[-1] if prev_unclear else 0) + (1 if unclear else 0)
        
        # Decrement liquid
        prev_liquid = st.session_state[f"{prefix}liquid_remaining_per_round"][s]
        last_liquid = prev_liquid[-1] if prev_liquid else 60_000
        new_liquid = max(0, last_liquid - LIQUID_PER_SHOT_UL)

        # For hunt_target, queue unclear cells for user resolution
        if unclear and s == "hunt_target":
            st.session_state[f"{prefix}unclear_queue"].append({
                "strategy": s,
                "row": row,
                "col": col,
                "score": score,
                "true_hit": is_hit_true,
                "round": round_num,
            })
            # Mark as previously unclear
            st.session_state[f"{prefix}previously_unclear_cells"][s].add((row, col))
            # Still execute with true value for now, but mark as needing review
            is_hit, sunk_ship = board.query(row, col)
            model.update(row, col, is_hit, sunk_ship)
            st.session_state[f"{prefix}cv_scores"][s][row, col] = score
            st.session_state[f"{prefix}binary_calls"][s][row, col] = binary_call
            st.session_state[f"{prefix}shot_order"][s].append((row, col))
            st.session_state[f"{prefix}cv_readings"][s].append(score)
            total_hits = board.total_ship_cells - board.get_remaining_ship_cells()
            st.session_state[f"{prefix}hits_per_round"][s].append(total_hits)
            st.session_state[f"{prefix}unclear_count_per_round"][s].append(cumulative_unclear)
            st.session_state[f"{prefix}liquid_remaining_per_round"][s].append(new_liquid)
        else:
            # normal flow
            is_hit, sunk_ship = board.query(row, col)
            model.update(row, col, is_hit, sunk_ship)
            st.session_state[f"{prefix}cv_scores"][s][row, col] = score
            st.session_state[f"{prefix}binary_calls"][s][row, col] = binary_call
            st.session_state[f"{prefix}shot_order"][s].append((row, col))
            st.session_state[f"{prefix}cv_readings"][s].append(score)
            total_hits = board.total_ship_cells - board.get_remaining_ship_cells()
            st.session_state[f"{prefix}hits_per_round"][s].append(total_hits)
            st.session_state[f"{prefix}unclear_count_per_round"][s].append(cumulative_unclear)
            st.session_state[f"{prefix}liquid_remaining_per_round"][s].append(new_liquid)

    st.session_state[f"{prefix}round"] = round_num

    if not any_active:
        st.session_state[f"{prefix}done"] = True


# ------------------------------------------------------------------
# Step logic: for failure-handling tab (pauses on unclear)
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Drawing helpers
# ------------------------------------------------------------------

def _draw_true_board(grid: np.ndarray, highlight_cell: tuple | None = None, small: bool = False, sunk_positions: set | None = None) -> matplotlib.figure.Figure:
    """Draw the ground-truth board with ship positions highlighted.
    
    Args:
        grid: The board grid
        highlight_cell: Optional (row, col) tuple to highlight with a bright border
        small: If True, create a smaller figure for dialog
        sunk_positions: Set of (row, col) tuples for cells that are part of sunk ships
    """
    if sunk_positions is None:
        sunk_positions = set()
    
    rows, cols = grid.shape
    figsize = (3.0, 2.2) if small else (6.5, 3.5)  # Match 96-well plate height
    fontsize_ticks = 5 if small else 6.5
    fontsize_title = 6.5 if small else 8
    fig, ax = plt.subplots(figsize=figsize)

    # Colour map: water=light blue, ship=dark red
    cmap = mcolors.ListedColormap(["#d4eaf7", "#c0392b"])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    # Grid lines
    for r in range(rows + 1):
        ax.axhline(r - 0.5, color="white", linewidth=1.2 if not small else 0.8)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color="white", linewidth=1.2 if not small else 0.8)

    # Overlay X on sunk ship cells
    for (r, c) in sunk_positions:
        ax.text(c, r, "X", ha="center", va="center", fontsize=14 if not small else 10, 
                fontweight="bold", color="white")

    # Highlight specific cell if provided
    if highlight_cell:
        hr, hc = highlight_cell
        rect = Rectangle((hc - 0.5, hr - 0.5), 1, 1,
                          linewidth=3 if small else 4, edgecolor='#f1c40f', facecolor='none')
        ax.add_patch(rect)
        # Add a glow effect
        rect2 = Rectangle((hc - 0.5, hr - 0.5), 1, 1,
                           linewidth=4 if small else 6, edgecolor='#f39c12', facecolor='none', alpha=0.5)
        ax.add_patch(rect2)

    ax.set_xticks(range(cols))
    ax.set_xticklabels([str(i) for i in range(1, cols + 1)], fontsize=fontsize_ticks)
    ax.set_yticks(range(rows))
    ax.set_yticklabels(ROW_LABELS[:rows], fontsize=fontsize_ticks)
    ax.set_title("True Board (red = ship)" if small else "True Board Layout  (red = ship / NaOH)", 
                 fontsize=fontsize_title, fontweight="bold")
    ax.tick_params(length=0, labelsize=fontsize_ticks - 1)

    fig.tight_layout()
    return fig


def _draw_96well_plate(
    cv_scores: np.ndarray,
    binary_calls: np.ndarray,
    shot_order: list,
    title: str | None = None, 
    show_scores: bool = True,
    game_over: bool = False,
    n_queries: int = 0,
    previously_unclear_cells: set | None = None,
) -> matplotlib.figure.Figure:
    """
    Draw a 96-well plate (8×12). Upper-left 8×10 grid is the game area;
    rightmost 2 columns are greyed out (unused).
    
    Args:
        cv_scores: Array of CV reading scores
        binary_calls: Array of binary calls (-2=not queried, -1=unclear, 0=hit, 1=miss)
        shot_order: List of (row, col) tuples in order of querying
        title: Optional title for the plate
        show_scores: Whether to display CV scores in wells
        game_over: If true, display game over message
        n_queries: Number of queries made
        previously_unclear_cells: Set of (row, col) tuples that were previously unclear
    """
    if previously_unclear_cells is None:
        previously_unclear_cells = set()
        
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.set_xlim(-0.6, PLATE_COLS - 0.4)
    ax.set_ylim(PLATE_ROWS - 0.4, -0.6)
    ax.set_aspect("equal")

    for r in range(PLATE_ROWS):
        for c in range(PLATE_COLS):
            if c >= BOARD_COLS:
                # unused columns
                circle = Circle((c, r), 0.4, facecolor="#e0e0e0",
                                    edgecolor="#bbb", linewidth=0.8)
                ax.add_patch(circle)
                continue

            bc = int(binary_calls[r, c])
            score_val = cv_scores[r, c]
            queried = not np.isnan(score_val)
            was_previously_unclear = (r, c) in previously_unclear_cells

            if not queried:
                fc = "#f5f5f5"
                ec = "#ccc"
                lw = 0.8
            elif bc == 0:       # hit
                fc = "#e74c3c"
                # If previously unclear, use yellow outline to indicate it was clarified by user
                ec = "#f1c40f" if was_previously_unclear else "#c0392b"
                lw = 2.0 if was_previously_unclear else 1.5
            elif bc == 1:       # miss
                fc = "#3498db"
                # If previously unclear, use yellow outline to indicate it was clarified by user
                ec = "#f1c40f" if was_previously_unclear else "#2980b9"
                lw = 2.0 if was_previously_unclear else 1.5
            elif bc == -1:      # unclear
                fc = "#f1c40f"
                ec = "#d4ac0f"
                lw = 2.0
            else:
                fc = "#f5f5f5"
                ec = "#ccc"
                lw = 0.8

            circle = Circle((c, r), 0.4, facecolor=fc, edgecolor=ec, linewidth=lw)
            ax.add_patch(circle)

            if queried and show_scores:
                ax.text(c, r, f"{score_val:.2f}", ha="center", va="center",
                        fontsize=4.5, fontweight="bold",
                        color="white" if bc in (0, 1) else "black")

            # Shot number
            if (r, c) in [(s[0], s[1]) for s in shot_order]:
                idx = next(i for i, s in enumerate(shot_order) if s == (r, c))
                ax.text(c + 0.3, r - 0.3, str(idx + 1), ha="center", va="center",
                        fontsize=3, color="#555")

    ax.set_xticks(range(PLATE_COLS))
    ax.set_xticklabels([str(i + 1) for i in range(PLATE_COLS)], fontsize=6)
    ax.set_yticks(range(PLATE_ROWS))
    ax.set_yticklabels(ROW_LABELS, fontsize=6)
    
    # Title with game over message
    if game_over:
        ax.set_title(f"All Ships Found! ({n_queries} shots)", fontsize=9, fontweight="bold", color="#27ae60")
    elif title:
        ax.set_title(title, fontsize=8.5, fontweight="bold")
    ax.tick_params(length=0, labelsize=5.5)

    # Legend
    try:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
                   markeredgecolor='#c0392b', markersize=8, label='Hit (ship)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
                   markeredgecolor='#2980b9', markersize=8, label='Miss (water)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
                   markeredgecolor='#f1c40f', linewidth=2, markersize=8, label='Clarified Hit (was unclear)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
                   markeredgecolor='#f1c40f', linewidth=2, markersize=8, label='Clarified Miss (was unclear)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#f1c40f',
                   markeredgecolor='#d4ac0f', markersize=8, label='Unclear'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#f5f5f5',
                   markeredgecolor='#ccc', markersize=8, label='Not queried'),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            fontsize=5,
            framealpha=0.9,
            handlelength=2.5,  # make handles longer for clarity
            borderpad=1.2,     # add more padding inside legend box
            labelspacing=1.0   # add more vertical space between entries
        )
    except Exception:
        pass  # Skip legend if there's any issue

    fig.tight_layout()
    return fig


def _draw_hits_chart(hits_per_round: dict, total_ship_cells: int = 17) -> matplotlib.figure.Figure:
    """Line chart: percentage of ship cells found vs round for all 4 strategies."""
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for s in STRATEGIES:
        data = hits_per_round[s]
        if data:
            pct = [100.0 * h / total_ship_cells for h in data]
            ax.plot(range(1, len(pct) + 1), pct, label=LABELS[s],
                    color=COLORS[s], linewidth=2)
    ax.set_xlabel("Round (shot number)", fontsize=8)
    ax.set_ylabel("% of Ship Cells Found", fontsize=8)
    ax.set_title("Percentage of Total Ship Cells Found", fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def _draw_unclear_cells_chart(unclear_count_per_round: dict) -> matplotlib.figure.Figure:
    """Line chart: cumulative # of unclear cells encountered vs round for all 4 strategies."""
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for s in STRATEGIES:
        data = unclear_count_per_round[s]
        if data:
            ax.plot(range(1, len(data) + 1), data, label=LABELS[s],
                    color=COLORS[s], linewidth=2, marker='o', markersize=4)
    ax.set_xlabel("Round (shot number)", fontsize=8)
    ax.set_ylabel("Cumulative # of Unclear Cells", fontsize=8)
    ax.set_title("Unclear Experimental Readings Encountered", fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def _draw_liquid_remaining_chart(liquid_remaining_per_round: dict) -> matplotlib.figure.Figure:
    """Line chart: remaining cabbage juice (mL) vs round for all 4 strategies."""
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for s in STRATEGIES:
        data = liquid_remaining_per_round[s]
        if data:
            # Convert from µL to mL
            data_mL = [val / 1000.0 for val in data]
            ax.plot(range(1, len(data_mL) + 1), data_mL, label=LABELS[s],
                    color=COLORS[s], linewidth=2, marker='o', markersize=4)
    ax.set_xlabel("Round (shot number)", fontsize=8)
    ax.set_ylabel("Cabbage Juice Remaining (mL)", fontsize=8)
    ax.set_title("Stopping Criteria: Liquid Remaining", fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='#c0392b', linestyle='--', linewidth=1.5, label='Empty', alpha=0.7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def _draw_variance_chart(cv_readings: dict) -> matplotlib.figure.Figure:
    """Line chart: running variance of CV readings per strategy."""
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for s in STRATEGIES:
        readings = cv_readings[s]
        if len(readings) > 1:
            # Calculate running variance (variance of all readings up to each point)
            variances = []
            for i in range(1, len(readings) + 1):
                var = np.var(readings[:i])
                variances.append(var)
            ax.plot(range(1, len(variances) + 1), variances, label=LABELS[s],
                    color=COLORS[s], linewidth=2, marker='o', markersize=4)
    ax.set_xlabel("Round (shot number)", fontsize=8)
    ax.set_ylabel("Variance of CV Readings", fontsize=8)
    ax.set_title("QC Metric: Variance of Experimental Readings", fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Save run results
# ------------------------------------------------------------------

def _save_run_results(prefix: str = "", tag: str = "simulation"):
    """Save visuals and a JSON log of the current run."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"{tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = st.session_state[f"{prefix}seed"]
    boards = st.session_state[f"{prefix}boards"]

    # True board (with sunk ship positions for 💥 overlay)
    sunk_positions = set()
    for ship in boards["hunt_target"].get_sunk_ships():
        sunk_positions.update(ship.positions)
    fig = _draw_true_board(boards["hunt_target"].grid, sunk_positions=sunk_positions)
    fig.savefig(run_dir / "true_board.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 96-well plate for hunt_target
    fig = _draw_96well_plate(
        st.session_state[f"{prefix}cv_scores"]["hunt_target"],
        st.session_state[f"{prefix}binary_calls"]["hunt_target"],
        st.session_state[f"{prefix}shot_order"]["hunt_target"],
        previously_unclear_cells=st.session_state[f"{prefix}previously_unclear_cells"]["hunt_target"],
    )
    fig.savefig(run_dir / "plate_hunt_target.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Hits chart
    fig = _draw_hits_chart(st.session_state[f"{prefix}hits_per_round"])
    fig.savefig(run_dir / "hits_vs_round.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Unclear cells chart
    fig = _draw_unclear_cells_chart(st.session_state[f"{prefix}unclear_count_per_round"])
    fig.savefig(run_dir / "unclear_vs_round.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # JSON summary
    summary = {
        "seed": seed,
        "total_rounds": st.session_state[f"{prefix}round"],
        "tag": tag,
        "timestamp": timestamp,
        "strategies": {},
    }
    for s in STRATEGIES:
        board = boards[s]
        summary["strategies"][s] = {
            "total_queries": board.n_queries,
            "game_over": board.is_game_over(),
            "hits_per_round": st.session_state[f"{prefix}hits_per_round"][s],
            "unclear_count_per_round": st.session_state[f"{prefix}unclear_count_per_round"][s],
            "shot_order": [(int(r), int(c)) for r, c in st.session_state[f"{prefix}shot_order"][s]],
        }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return run_dir


# ==================================================================
# Streamlit App
# ==================================================================

def _inject_global_css() -> None:
    st.markdown(
        """
        <style>
            .main { padding-top: 1rem; }
            h2, h3 { margin-top: 0.5rem; margin-bottom: 0.3rem; }
            .element-container { margin-bottom: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_home() -> None:
    _inject_global_css()
    st.title("BaiTu — Battleship Active Learning on an OT-2 Liquid Handler")
    st.markdown(
        """
BaiTu treats a game of Battleship as a proxy for an automated wet-lab
experiment. The Opentrons OT-2 dispenses **NaOH** into ship positions and
**water** into empty positions on a 96-well plate. Each round, cabbage-juice
indicator is added to a queried well — it turns **purple** for a hit (NaOH)
or **blue** for a miss (water). A computer-vision readout feeds the result
back into a Bayesian belief model that picks the next well to query.

Use the sidebar to navigate:

| Page | What it does |
| --- | --- |
| **🧪 Simulator** | Run all four acquisition strategies in the browser. No hardware required. Handles unclear CV readings with a human-in-the-loop prompt. |
| **🤖 OT-2 Hardware** | Configure, launch, pause / resume, stop, and reset a real OT-2 closed-loop run (or a dry-run that exercises the full pipeline without hardware). Live plate state, belief map, camera frames, and streaming logs. |

---

### What each page uses under the hood

- The simulator re-implements the game loop in `streamlit` using
  `core.battleship_env` + `core.battleship_model` directly, so interactions
  are instant.
- The hardware page spawns `python -m hardware.battleship_ot2_loop` as a
  subprocess. **Stop** sends `SIGTERM` (equivalent to pressing Ctrl-C on the
  CLI), and **Reset** shells out to
  `python -m hardware.battleship_ot2_loop reset --robot_ip …` to home the
  pipette and drop any attached tip.
"""
    )


def render_simulator() -> None:
    _inject_global_css()
    st.title("🧪 Simulator")
    st.caption(
        "In-browser Battleship simulation with four acquisition strategies. "
        "Unclear CV readings pause for a human hit / miss call."
    )
    _render_simulation_tab(prefix="")


def render_hardware() -> None:
    _inject_global_css()
    st.title("🤖 OT-2 Hardware")
    st.caption(
        "Drives `hardware.battleship_ot2_loop` in a subprocess — Stop sends "
        "SIGTERM like a manual Ctrl-C. Enable **Dry-run** to rehearse without "
        "hardware."
    )
    _render_hardware_tab()


def main() -> None:
    st.set_page_config(
        page_title="BaiTu",
        page_icon="🧪",
        layout="wide",
    )
    pages = [
        st.Page(render_home, title="Home", icon="🏠", default=True),
        st.Page(render_simulator, title="Simulator", icon="🧪"),
        st.Page(render_hardware, title="OT-2 Hardware", icon="🤖"),
    ]
    st.navigation(pages).run()


# ------------------------------------------------------------------
# Tab 1: Simulation
# ------------------------------------------------------------------

def _render_simulation_tab(prefix: str = ""):
    _init_session(prefix)

    boards = st.session_state[f"{prefix}boards"]
    done = st.session_state[f"{prefix}done"]
    current_round = st.session_state[f"{prefix}round"]
    unclear_queue = st.session_state[f"{prefix}unclear_queue"]

    # --- Process pending dialog response (close dialog first, then continue) ---
    pending = st.session_state.get(f"{prefix}dialog_response", None)
    if pending is not None:
        # 1. Apply the saved response to session state
        st.session_state[f"{prefix}binary_calls"]["hunt_target"][pending["row"], pending["col"]] = pending["value"]
        if unclear_queue and unclear_queue[0]["row"] == pending["row"] and unclear_queue[0]["col"] == pending["col"]:
            st.session_state[f"{prefix}unclear_queue"].pop(0)
        st.session_state[f"{prefix}dialog_response"] = None
        # 2. Inject JS to force-close the dialog
        components.html("""
        <script>
        var closeBtn = window.parent.document.querySelector(
            '[data-testid="stDialog"] button[aria-label="Close"]'
        );
        if (closeBtn) closeBtn.click();
        </script>
        """, height=0, width=0)
        # 3. Rerun to reflect the cleared dialog and updated state
        st.rerun()

    # Check if paused for unclear cell (blocks auto-play)
    unclear_queue = st.session_state[f"{prefix}unclear_queue"]
    paused = len(unclear_queue) > 0

    # Auto-play mode is derived from session state
    auto_play = st.session_state.get(f"{prefix}auto_play", False)

    # --- Modal overlay for unclear cells ---
    if paused and len(unclear_queue) > 0:
        # If we are in auto-play, pause it and flag to resume later
        if auto_play:
            st.session_state[f"{prefix}resume_auto_play"] = True
            st.session_state[f"{prefix}auto_play"] = False
            auto_play = False # Update local var


        item = unclear_queue[0]
        row_label = ROW_LABELS[item["row"]]
        col_label = item["col"] + 1
        item_key = f"{item['row']}_{item['col']}"
        
        # Get the true board grid and sunk ship positions for display in dialog
        grid = boards["hunt_target"].grid
        dialog_sunk_positions = set()
        for ship in boards["hunt_target"].get_sunk_ships():
            dialog_sunk_positions.update(ship.positions)

        @st.dialog(f"⚠️ Unclear Reading - Well {row_label}{col_label}")
        def unclear_dialog():
            col_board, col_info = st.columns([1, 1])
            
            with col_board:
                fig_board = _draw_true_board(grid, highlight_cell=(item["row"], item["col"]), small=True, sunk_positions=dialog_sunk_positions)
                st.pyplot(fig_board, width="stretch")
                plt.close(fig_board)
            
            with col_info:
                st.markdown(f"""
                <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%; min-height: 180px;">
                    <div style="font-size: 1.3rem; font-weight: bold; margin-bottom: 0.5rem; color: #e67e22;">
                        Well {row_label}{col_label}
                    </div>
                    <div style="font-size: 1rem; color: #666; margin-bottom: 0.8rem;">
                        CV Score: {item['score']:.3f}
                    </div>
                    <div style="font-size: 0.95rem;">
                        Is this a <strong>HIT</strong> or <strong>MISS</strong>?
                    </div>
                </div>
                """, unsafe_allow_html=True)

            col_hit, col_miss = st.columns(2)
            with col_hit:
                if st.button("🎯 HIT (Ship)", key=f"{prefix}dialog_hit_{item_key}", width="stretch", type="primary"):
                    # Save response — will be processed on next rerun
                    st.session_state[f"{prefix}dialog_response"] = {
                        "row": item["row"], "col": item["col"], "value": 0
                    }
                    st.rerun()
            with col_miss:
                if st.button("💧 MISS (Water)", key=f"{prefix}dialog_miss_{item_key}", width="stretch", type="primary"):
                    st.session_state[f"{prefix}dialog_response"] = {
                        "row": item["row"], "col": item["col"], "value": 1
                    }
                    st.rerun()

        unclear_dialog()
        
    # --- Strategy selector buttons (inline, before visuals) ---
    # --- Strategy selector buttons with status indicators ---
    st.markdown("**View 96-Well Plate:**")
    col_strat1, col_strat2, col_strat3, col_strat4, col_spacer = st.columns([1, 1, 1, 1, 2])
    
    current_display = st.session_state[f"{prefix}display_strategy"]
    
    strategies_list = [("random", "Random"), ("prob", "Probability"), ("entropy", "Entropy"), ("hunt_target", "Hunt-Target")]
    cols = [col_strat1, col_strat2, col_strat3, col_strat4]
    
    for col, (strategy, label) in zip(cols, strategies_list):
        board = boards[strategy]
        is_active = current_display == strategy
        is_done = board.is_game_over()
        
        # Simple label with checkmark if done, otherwise just the name
        if is_done:
            btn_label = f"✅ {label}"
        else:
            btn_label = label
        
        with col:
            clicked = st.button(btn_label, key=f"{prefix}strat_{strategy}", width="stretch", 
                              type="primary" if is_active else "secondary")
            if clicked:
                st.session_state[f"{prefix}display_strategy"] = strategy
                st.rerun()

    # --- ROW 1: True Board (40%) | 96-Well Plate (60%) ---
    col_board, col_plate = st.columns([0.4, 0.6])

    with col_board:
        selected_strat = st.session_state[f"{prefix}display_strategy"]
        selected_board = boards[selected_strat]
        # Get sunk ship positions for the selected strategy
        sunk_positions = set()
        for ship in selected_board.get_sunk_ships():
            sunk_positions.update(ship.positions)
        st.markdown(f"### True Board – {LABELS[selected_strat]}")
        fig_board = _draw_true_board(selected_board.grid, sunk_positions=sunk_positions)
        st.pyplot(fig_board, width="stretch")
        plt.close(fig_board)

    with col_plate:
        selected_strat = st.session_state[f"{prefix}display_strategy"]
        selected_board = boards[selected_strat]
        strat_game_over = selected_board.is_game_over()
        strat_n_queries = selected_board.n_queries
        sunk_ship_sizes = [ship.size for ship in selected_board.get_sunk_ships()]
        
        st.markdown(f"### 96-Well Plate – {LABELS[selected_strat]}")
        fig_plate = _draw_96well_plate(
            st.session_state[f"{prefix}cv_scores"][selected_strat],
            st.session_state[f"{prefix}binary_calls"][selected_strat],
            st.session_state[f"{prefix}shot_order"][selected_strat],
            title=LABELS[selected_strat],
            game_over=strat_game_over,
            n_queries=strat_n_queries,
            previously_unclear_cells=st.session_state[f"{prefix}previously_unclear_cells"][selected_strat],
        )
        st.pyplot(fig_plate, width="stretch")
        plt.close(fig_plate)

    # --- Control buttons (under the visuals) ---
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    with col1:
        if st.button("🆕 New Board", key=f"{prefix}btn_new_board", width="stretch"):
            seed = st.session_state[f"{prefix}next_seed"]
            _reset_session(seed, prefix)
            st.session_state[f"{prefix}next_seed"] = seed + 1
            st.rerun()

    with col2:
        step_disabled = done or paused or auto_play
        if st.button("▶️ Next Shot", key=f"{prefix}btn_step", disabled=step_disabled, width="stretch"):
            _step_all_strategies(prefix)
            st.rerun()

    with col3:
        if st.button("⏩ Play 5 Shots", key=f"{prefix}btn_auto", disabled=(done or paused or auto_play), width="stretch"):
            for _ in range(5):
                if not st.session_state[f"{prefix}done"] and len(st.session_state[f"{prefix}unclear_queue"]) == 0:
                    _step_all_strategies(prefix)
                    if len(st.session_state[f"{prefix}unclear_queue"]) > 0:
                        break  # pause on unclear
            st.rerun()

    with col4:
        if auto_play:
            # Show Stop button when auto-playing
            if st.button("⏹️ Stop", key=f"{prefix}btn_stop", width="stretch", type="primary"):
                st.session_state[f"{prefix}auto_play"] = False
                st.rerun()
        else:
            if st.button("▶️▶️ Play All", key=f"{prefix}btn_play_all", disabled=(done or paused), width="stretch"):
                st.session_state[f"{prefix}auto_play"] = True
                st.rerun()

    with col5:
        if st.button("💾 Save", key=f"{prefix}btn_save", width="stretch"):
            run_dir = _save_run_results(prefix, tag="simulation")
            st.success(f"Saved to {run_dir}")

    # --- Resume auto-play via JS after dialog was closed ---
    # Placed AFTER the Play All button is rendered so it exists in the DOM
    resume_auto_play = st.session_state.get(f"{prefix}resume_auto_play", False)
    if resume_auto_play and not paused and not done:
        st.session_state[f"{prefix}resume_auto_play"] = False
        components.html("""
        <script>
        (function() {
            // Wait 1s for Streamlit to fully finish rendering,
            // then find and click the Play All button
            setTimeout(function() {
                var buttons = window.parent.document.querySelectorAll('button');
                for (var i = 0; i < buttons.length; i++) {
                    if (buttons[i].innerText.indexOf('Play All') !== -1
                        && !buttons[i].disabled) {
                        buttons[i].click();
                        return;
                    }
                }
            }, 1000);
        })();
        </script>
        """, height=0, width=0)

    st.markdown("---")

    # --- ROW 1: Stopping Criteria (left: Liquid Remaining | right: % Ships Found) ---
    st.markdown("### Stopping Criteria")
    col_liquid, col_ships = st.columns(2)
    
    with col_liquid:
        fig_liquid = _draw_liquid_remaining_chart(st.session_state[f"{prefix}liquid_remaining_per_round"])
        st.pyplot(fig_liquid, width="stretch")
        plt.close(fig_liquid)
    
    with col_ships:
        total_cells = boards["hunt_target"].total_ship_cells
        fig_hits = _draw_hits_chart(st.session_state[f"{prefix}hits_per_round"], total_cells)
        st.pyplot(fig_hits, width="stretch")
        plt.close(fig_hits)

    st.markdown("---")

    # --- ROW 2: QC Metrics (left: Variance | right: Unclear Cells) ---
    st.markdown("### QC Metrics")
    col_variance, col_unclear = st.columns(2)
    
    with col_variance:
        fig_variance = _draw_variance_chart(st.session_state[f"{prefix}cv_readings"])
        st.pyplot(fig_variance, width="stretch")
        plt.close(fig_variance)
    
    with col_unclear:
        fig_unclear = _draw_unclear_cells_chart(st.session_state[f"{prefix}unclear_count_per_round"])
        st.pyplot(fig_unclear, width="stretch")
        plt.close(fig_unclear)

    st.markdown("---")

    # --- ROW 3: Strategy Summary (full width, always visible) ---
    st.markdown("### Strategy Summary")
    summary_data = []
    for s in STRATEGIES:
        board = boards[s]
        unclear_list = st.session_state[f"{prefix}unclear_count_per_round"][s]
        liquid_list = st.session_state[f"{prefix}liquid_remaining_per_round"][s]
        current_unclear = unclear_list[-1] if unclear_list else 0
        current_liquid = liquid_list[-1] / 1000 if liquid_list else 0  # convert µL to mL
        summary_data.append({
            "Strategy": LABELS[s],
            "Shots": board.n_queries,
            "Unclear Readings": current_unclear,
            "Liquid Left (mL)": f"{current_liquid:.2f}",
            "Done": "✅" if board.is_game_over() else "—",
        })
    st.table(summary_data)

    # --- Auto-play continuation (at end so visuals render first) ---
    if auto_play and not paused and not done:
        _step_all_strategies(prefix)
        time.sleep(0.4)  # 400ms delay between shots for visible updates
        st.rerun()


# ------------------------------------------------------------------
# Tab 2: Handle Logical Failures (deprecated – merged into main tab)
# ------------------------------------------------------------------
# (This functionality is now integrated into the main simulation tab
#  with a modal dialog for unclear cells)


# ==================================================================
# Tab: OT-2 Hardware Run
# ==================================================================

_HW_PHASE_BADGE = {
    "idle":    ("⚪ Idle",    "#6b7280"),
    "setup":   ("🛠️ Setup",   "#3b82f6"),
    "loop":    ("▶️ Running", "#10b981"),
    "paused":  ("⏸ Paused",   "#f59e0b"),
    "stopped": ("⏹ Stopped",  "#ef4444"),
    "done":    ("✅ Done",     "#10b981"),
    "error":   ("❌ Error",    "#ef4444"),
}

_DEFAULT_GEOMETRY_PATH = "hardware/calibration.json"


def _get_ot2_controller() -> OT2Controller:
    ctrl = st.session_state.get("ot2_ctrl")
    if ctrl is None:
        ctrl = OT2Controller()
        st.session_state["ot2_ctrl"] = ctrl
    return ctrl


@st.cache_resource
def _get_live_preview(device_index: int = 0) -> LivePreview:
    """One shared LivePreview handle per Streamlit server process."""
    return LivePreview(device_index=device_index)


def _plot_board_heatmap(
    matrix: np.ndarray,
    title: str,
    *,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotate_values: bool = False,
) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([str(c + 1) for c in range(matrix.shape[1])], fontsize=7)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(list("ABCDEFGH")[: matrix.shape[0]], fontsize=7)
    ax.set_title(title, fontsize=10)
    if annotate_values:
        # Pick text colour per-cell based on its normalised intensity so
        # values remain legible on both dark and bright cmap regions.
        v_lo = vmin if vmin is not None else float(np.nanmin(matrix))
        v_hi = vmax if vmax is not None else float(np.nanmax(matrix))
        span = max(v_hi - v_lo, 1e-9)
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                v = matrix[r, c]
                if np.isnan(v):
                    continue
                norm = (v - v_lo) / span
                txt_colour = "white" if norm < 0.55 else "black"
                txt = f"{v:.2f}" if isinstance(v, float) or np.issubdtype(matrix.dtype, np.floating) else f"{int(v)}"
                ax.text(c, r, txt, ha="center", va="center",
                        fontsize=6, color=txt_colour)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    return fig


def _plot_results_plate(matrix_list: List[List[int]]) -> matplotlib.figure.Figure:
    """Render an 8×12 plate with hit / miss / unprobed styling."""
    mat = np.asarray(matrix_list, dtype=int)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    rows, cols = mat.shape
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks(range(cols))
    ax.set_xticklabels([str(c + 1) for c in range(cols)], fontsize=7)
    ax.set_yticks(range(rows))
    ax.set_yticklabels(list("ABCDEFGH")[:rows], fontsize=7)
    ax.set_aspect("equal")
    ax.set_title("Plate state — hits (purple), misses (blue), unprobed (grey)", fontsize=10)

    for r in range(rows):
        for c in range(cols):
            v = mat[r, c]
            if c >= BOARD_COLS:
                color = "#f3f4f6"   # reserved control columns
            elif v == 1:
                color = "#a855f7"   # hit → NaOH turned purple
            elif v == 0:
                color = "#3b82f6"   # miss → water stays blue
            else:
                color = "#d1d5db"   # unprobed
            ax.add_patch(Circle((c, r), 0.38, color=color, ec="#111827", lw=0.5))
    fig.tight_layout()
    return fig


_DECK_SLOT_COLOURS = {
    "plate":     ("#a855f7", "Plate"),           # purple
    "tiprack":   ("#f59e0b", "Tiprack"),         # amber
    "naoh":      ("#3b82f6", "NaOH"),            # blue
    "h2o":       ("#22d3ee", "H₂O"),             # cyan
    "indicator": ("#ec4899", "Indicator"),       # pink
}
# OT-2 physical slot grid: top row = slots 10/11/trash, bottom row = 1/2/3.
_DECK_ROWS = [
    ["10", "11", "trash"],
    ["7",  "8",  "9"],
    ["4",  "5",  "6"],
    ["1",  "2",  "3"],
]


def _effective_deck() -> Dict[str, object]:
    """Defaults merged with any session-state overrides."""
    overrides = st.session_state.get("hw_deck_overrides", {}) or {}
    return {**DEFAULT_DECK, **overrides}


def _deck_occupants(deck: Dict[str, object]) -> Dict[str, Dict[str, str]]:
    """Map slot-string → {kind, title, labware, extra} for visualisation."""
    occ: Dict[str, Dict[str, str]] = {}
    pairs = [
        ("plate",     "plate_slot",     "plate_labware",     None),
        ("tiprack",   "tiprack_slot",   "tiprack_labware",   None),
        ("naoh",      "naoh_slot",      "naoh_labware",      "naoh_source_well"),
        ("h2o",       "h2o_slot",       "h2o_labware",       "h2o_source_well"),
        ("indicator", "indicator_slot", "indicator_labware", "indicator_source_well"),
    ]
    for kind, slot_k, labware_k, well_k in pairs:
        slot = str(deck.get(slot_k, ""))
        if not slot:
            continue
        colour, title = _DECK_SLOT_COLOURS[kind]
        occ[slot] = {
            "kind":    kind,
            "title":   title,
            "colour":  colour,
            "labware": str(deck.get(labware_k, "")),
            "extra":   f"well {deck[well_k]}" if well_k and deck.get(well_k) else "",
        }
    return occ


def _render_deck_panel() -> None:
    """Draw the current OT-2 deck layout and an editable override form."""
    deck = _effective_deck()
    occ = _deck_occupants(deck)
    has_overrides = bool(st.session_state.get("hw_deck_overrides"))

    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.markdown("#### Deck layout")
        pip = f"{deck['pipette_name']} on **{deck['pipette_mount']}** mount"
        fill_v = float(deck["fill_volume"])
        ind_v = float(deck["indicator_volume"])
        st.caption(
            f"Pipette: {pip} · Fill volume: {fill_v:.0f} µL · "
            f"Indicator volume: {ind_v:.0f} µL"
            + ("  —  *(overrides active)*" if has_overrides else "")
        )
    with header_cols[1]:
        if has_overrides and st.button("Reset to defaults", use_container_width=True,
                                       key="hw_deck_reset"):
            st.session_state["hw_deck_overrides"] = {}
            st.rerun()

    # ── 4 × 3 deck grid ───────────────────────────────────────────────
    for row in _DECK_ROWS:
        cols = st.columns(3)
        for i, slot in enumerate(row):
            with cols[i]:
                if slot == "trash":
                    _slot_card("trash", "Trash", "#6b7280", "", "")
                else:
                    info = occ.get(slot)
                    if info is None:
                        _slot_card(slot, "", "#e5e7eb", "empty", "")
                    else:
                        _slot_card(
                            slot, info["title"], info["colour"],
                            info["labware"], info["extra"],
                        )

    # ── Editable form ─────────────────────────────────────────────────
    with st.expander("Edit deck layout (applied to the next run)", expanded=False):
        st.caption(
            "Defaults live in `hardware/battleship_ot2_loop.py::DEFAULT_DECK`. "
            "Edits here override them for the next **Start**; the override is "
            "saved to `<output_dir>/deck.json` and passed via `--deck_path`."
        )

        with st.form("hw_deck_form", clear_on_submit=False):
            pipette_cols = st.columns(2)
            with pipette_cols[0]:
                pipette_name = st.text_input(
                    "Pipette name", value=str(deck["pipette_name"]),
                    help="Opentrons model name, e.g. `p1000_single` or `p300_single_gen2`.",
                )
            with pipette_cols[1]:
                pipette_mount = st.selectbox(
                    "Pipette mount", options=["left", "right"],
                    index=0 if deck["pipette_mount"] == "left" else 1,
                )

            st.markdown("**Slot assignments**")
            slot_opts = [str(i) for i in range(1, 12)]

            def _slot_select(label: str, key: str) -> str:
                current = str(deck[key])
                return st.selectbox(
                    label, options=slot_opts,
                    index=slot_opts.index(current) if current in slot_opts else 0,
                    key=f"hw_deck_form_{key}",
                )

            r1 = st.columns(2)
            with r1[0]:
                plate_slot = _slot_select("Plate slot", "plate_slot")
                plate_labware = st.text_input("Plate labware", value=str(deck["plate_labware"]))
            with r1[1]:
                tiprack_slot = _slot_select("Tiprack slot", "tiprack_slot")
                tiprack_labware = st.text_input("Tiprack labware", value=str(deck["tiprack_labware"]))

            r2 = st.columns(3)
            with r2[0]:
                st.markdown("*NaOH reservoir*")
                naoh_slot = _slot_select("Slot", "naoh_slot")
                naoh_labware = st.text_input("Labware", value=str(deck["naoh_labware"]),
                                             key="hw_deck_form_naoh_labware")
                naoh_well = st.text_input("Source well", value=str(deck["naoh_source_well"]),
                                          key="hw_deck_form_naoh_well")
            with r2[1]:
                st.markdown("*H₂O reservoir*")
                h2o_slot = _slot_select("Slot", "h2o_slot")
                h2o_labware = st.text_input("Labware", value=str(deck["h2o_labware"]),
                                            key="hw_deck_form_h2o_labware")
                h2o_well = st.text_input("Source well", value=str(deck["h2o_source_well"]),
                                         key="hw_deck_form_h2o_well")
            with r2[2]:
                st.markdown("*Indicator reservoir*")
                indicator_slot = _slot_select("Slot", "indicator_slot")
                indicator_labware = st.text_input("Labware", value=str(deck["indicator_labware"]),
                                                  key="hw_deck_form_indicator_labware")
                indicator_well = st.text_input("Source well",
                                               value=str(deck["indicator_source_well"]),
                                               key="hw_deck_form_indicator_well")

            vol_cols = st.columns(2)
            with vol_cols[0]:
                fill_volume = st.number_input(
                    "Fill volume per well (µL)",
                    min_value=10.0, max_value=1000.0, step=10.0,
                    value=float(deck["fill_volume"]),
                )
            with vol_cols[1]:
                indicator_volume = st.number_input(
                    "Indicator volume per well (µL)",
                    min_value=10.0, max_value=1000.0, step=10.0,
                    value=float(deck["indicator_volume"]),
                )

            submitted = st.form_submit_button("Apply deck changes", type="primary")

        if submitted:
            proposed = {
                "pipette_name":          pipette_name.strip() or DEFAULT_DECK["pipette_name"],
                "pipette_mount":         pipette_mount,
                "plate_slot":            plate_slot,
                "plate_labware":         plate_labware.strip() or DEFAULT_DECK["plate_labware"],
                "tiprack_slot":          tiprack_slot,
                "tiprack_labware":       tiprack_labware.strip() or DEFAULT_DECK["tiprack_labware"],
                "naoh_slot":             naoh_slot,
                "naoh_labware":          naoh_labware.strip() or DEFAULT_DECK["naoh_labware"],
                "naoh_source_well":      naoh_well.strip() or DEFAULT_DECK["naoh_source_well"],
                "h2o_slot":              h2o_slot,
                "h2o_labware":           h2o_labware.strip() or DEFAULT_DECK["h2o_labware"],
                "h2o_source_well":       h2o_well.strip() or DEFAULT_DECK["h2o_source_well"],
                "indicator_slot":        indicator_slot,
                "indicator_labware":     indicator_labware.strip() or DEFAULT_DECK["indicator_labware"],
                "indicator_source_well": indicator_well.strip() or DEFAULT_DECK["indicator_source_well"],
                "fill_volume":           float(fill_volume),
                "indicator_volume":      float(indicator_volume),
            }
            # Validate: no two roles may share the same slot.
            slot_assignments = [
                ("plate", proposed["plate_slot"]),
                ("tiprack", proposed["tiprack_slot"]),
                ("NaOH", proposed["naoh_slot"]),
                ("H₂O", proposed["h2o_slot"]),
                ("indicator", proposed["indicator_slot"]),
            ]
            seen: Dict[str, str] = {}
            clashes: List[str] = []
            for role, slot in slot_assignments:
                if slot in seen:
                    clashes.append(f"slot {slot}: {seen[slot]} + {role}")
                seen[slot] = role
            if clashes:
                st.error("Slot conflict — " + "; ".join(clashes))
            else:
                overrides = {k: v for k, v in proposed.items() if v != DEFAULT_DECK[k]}
                st.session_state["hw_deck_overrides"] = overrides
                st.success(
                    "Deck updated."
                    if overrides
                    else "All values match the defaults — no override stored."
                )
                st.rerun()


def _slot_card(slot: str, title: str, colour: str, labware: str, extra: str) -> None:
    """Render one deck-slot card with colour accent and label stack."""
    body = (
        f"<div style='border-left:4px solid {colour}; background:{colour}14; "
        f"border-radius:0.4rem; padding:0.5rem 0.65rem; margin:0.15rem 0; "
        f"min-height:76px;'>"
        f"<div style='font-size:0.75rem; color:#6b7280;'>Slot {slot}</div>"
        f"<div style='font-weight:600; color:{colour}; font-size:0.95rem;'>"
        f"{title or '&nbsp;'}</div>"
        f"<div style='font-size:0.72rem; color:#374151; "
        f"word-break:break-all; line-height:1.1;'>{labware or '&nbsp;'}</div>"
    )
    if extra:
        body += f"<div style='font-size:0.7rem; color:#6b7280;'>{extra}</div>"
    body += "</div>"
    st.markdown(body, unsafe_allow_html=True)


def _render_live_preview_panel(running: bool) -> bool:
    """Show a toggle + live webcam frame. Returns whether the preview is on."""
    st.markdown("**Live preview**")
    # When a run is active, the subprocess opens the same camera device for
    # each step. Default the preview OFF in that state to avoid contention.
    default_on = not running
    if "hw_live_camera" not in st.session_state:
        st.session_state["hw_live_camera"] = default_on

    col_toggle, col_dev = st.columns([2, 1])
    with col_toggle:
        live_on = st.toggle(
            "Stream camera",
            key="hw_live_camera",
            help=(
                "Grabs a frame from cv2.VideoCapture on every rerun (~1 Hz). "
                "Webcams are exclusive — while the OT-2 loop is running the "
                "subprocess also needs this device, so leaving the preview on "
                "during a run may cause occasional step-capture retries."
            ),
        )
    with col_dev:
        device_idx = int(st.number_input(
            "Device", min_value=0, max_value=8,
            value=int(st.session_state.get("hw_live_camera_device", 0)),
            step=1, key="hw_live_camera_device",
            label_visibility="collapsed",
            help="Webcam device index (0 = default)",
        ))

    cam = _get_live_preview(device_idx)

    if not live_on:
        cam.release()
        st.caption("Preview off. Toggle on to see the live webcam.")
        return False

    if running:
        st.caption("⚠ Sharing the camera with the running subprocess.")

    if not cam.open():
        st.warning(cam.last_error() or "Could not open camera.")
        return live_on

    frame = cam.grab()
    if frame is None:
        st.caption(cam.last_error() or "Waiting for first frame …")
        return live_on

    ts = datetime.datetime.fromtimestamp(cam.last_frame_ts()).strftime("%H:%M:%S")
    st.image(frame, caption=f"Live @ {ts}", width="stretch")
    return live_on


def _render_human_check_panel(ctrl: "OT2Controller", snap) -> None:
    """Banner shown when the subprocess is blocked on an ambiguous reading."""
    req = snap.human_check_request or {}
    well = req.get("well", "?")
    auto_label = req.get("auto_label", "?")
    conf = float(req.get("confidence", 0.5))
    rgb = req.get("mean_rgb", [0, 0, 0])
    img_path = req.get("image_path")

    st.markdown(
        f"<div style='padding:0.6rem 0.9rem; border-radius:0.5rem; "
        f"background:#fef3c7; border-left:4px solid #f59e0b; margin:0.4rem 0;'>"
        f"<b>⚠ Human check needed.</b> Reading for well <b>{well}</b> is ambiguous "
        f"(auto={auto_label}, conf={conf:.3f}, RGB={tuple(int(x) for x in rgb)}). "
        f"The OT-2 loop is paused — please confirm hit or miss."
        f"</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns([1, 1, 1])
    with cols[0]:
        if img_path and Path(img_path).exists():
            st.image(img_path, caption=Path(img_path).name, width="stretch")
        else:
            st.caption("(no captured image)")
    with cols[1]:
        if st.button("🎯 HIT (ship)", key="hw_hcheck_hit",
                     type="primary", width="stretch"):
            ctrl.submit_human_check("ship")
            st.rerun()
    with cols[2]:
        if st.button("💧 MISS (water)", key="hw_hcheck_miss",
                     type="primary", width="stretch"):
            ctrl.submit_human_check("water")
            st.rerun()


def _render_manual_correction_panel(ctrl: "OT2Controller", snap) -> None:
    """Operator override: relabel a previously-classified well."""
    history = snap.history or []
    if not history:
        return

    pending_n = snap.pending_corrections or 0
    expander_title = "✏ Edit / correct wells"
    if pending_n:
        expander_title += f"  ·  {pending_n} pending"

    with st.expander(expander_title, expanded=False):
        st.caption(
            "Select a previously-classified well and override its label. "
            "Corrections are queued and applied by the OT-2 loop before the "
            "next acquisition step (the belief model is rebuilt to reflect "
            "the corrected history)."
        )

        # Sort newest first so recent (and most likely wrong) calls are easy to find
        options = []
        for rec in reversed(history):
            options.append({
                "label": (
                    f"{rec['well_name']} — auto={rec['label']} "
                    f"(conf={float(rec['confidence']):.2f}, step={rec['step']})"
                ),
                "row": int(rec["row"]),
                "col": int(rec["col"]),
                "current": rec["label"],
            })

        idx = st.selectbox(
            "Well",
            options=list(range(len(options))),
            format_func=lambda i: options[i]["label"],
            key="hw_correct_well_idx",
        )
        sel = options[int(idx)]

        new_label = st.radio(
            "Correct label",
            options=["ship", "water"],
            horizontal=True,
            index=0 if sel["current"] == "ship" else 1,
            key="hw_correct_label",
        )

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("Queue correction", key="hw_correct_submit",
                         type="primary", width="stretch"):
                if new_label == sel["current"]:
                    st.info("Label unchanged — nothing to queue.")
                else:
                    ok = ctrl.queue_correction(sel["row"], sel["col"], new_label)
                    if ok:
                        st.success(
                            f"Queued: {ROW_LABELS[sel['row']]}{sel['col']+1} → {new_label}. "
                            "The OT-2 loop will rebuild its belief on the next step."
                        )
                    else:
                        st.error("Could not queue correction (no active run).")
        with c2:
            if st.button("Clear pending", key="hw_correct_clear", width="stretch",
                         disabled=pending_n == 0):
                ctrl.clear_pending_corrections()
                st.rerun()


def _render_hardware_tab() -> None:
    ctrl = _get_ot2_controller()
    running = ctrl.is_running()
    snap = ctrl.snapshot()

    # ── Deck layout + editor ──────────────────────────────────────────
    _render_deck_panel()

    # ── Configuration form ────────────────────────────────────────────
    with st.expander("Experiment configuration", expanded=not running):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            strategy = st.selectbox(
                "Strategy",
                ["prob", "entropy", "hunt_target", "pro_solver", "random"],
                index=0,
                key="hw_strategy",
            )
            dry_run = st.checkbox("Dry-run (no hardware)", value=True, key="hw_dry_run")
            skip_setup = st.checkbox(
                "Skip setup (plate already prepared)",
                value=False, key="hw_skip_setup",
            )
        with col_b:
            seed = st.number_input("Seed", min_value=0, max_value=10_000,
                                   value=42, step=1, key="hw_seed")
            robot_ip = st.text_input("Robot IP", value="169.254.200.128",
                                     key="hw_robot_ip")
            develop_s = st.slider(
                "Colour develop (s)",
                min_value=0.0, max_value=60.0, value=10.0, step=1.0,
                key="hw_develop_s",
            )
        with col_c:
            geometry_path = st.text_input(
                "Calibration JSON",
                value=_DEFAULT_GEOMETRY_PATH,
                help="Path to calibration.json with geometry + RGB prototypes",
                key="hw_geometry_path",
            )
            default_out = datetime.datetime.now().strftime("ot2_loop_%Y%m%d_%H%M%S")
            output_dir = st.text_input(
                "Output directory", value=default_out, key="hw_output_dir",
            )

    # ── Status badge ──────────────────────────────────────────────────
    label, colour = _HW_PHASE_BADGE.get(snap.phase, ("?", "#6b7280"))
    msg = snap.phase_message or ""
    st.markdown(
        f"<div style='padding:0.5rem 0.85rem; border-radius:0.5rem; "
        f"background:{colour}20; color:{colour}; font-weight:600; margin:0.2rem 0 0.6rem;'>"
        f"{label} — <span style='font-weight:400;'>{msg}</span></div>",
        unsafe_allow_html=True,
    )

    # ── Control bar: Start | Pause/Resume | Stop | Reset ──────────────
    ctrl_cols = st.columns(4)
    with ctrl_cols[0]:
        start_pressed = st.button(
            "▶ Start", disabled=running, width="stretch", type="primary",
        )
    with ctrl_cols[1]:
        posix = os.name == "posix"
        if ctrl.is_paused():
            toggle_pressed = st.button("⏵ Resume", width="stretch")
            toggle_is_resume = True
        else:
            toggle_pressed = st.button(
                "⏸ Pause",
                disabled=not running or not posix,
                width="stretch",
                help=None if posix else "Pause is only supported on POSIX.",
            )
            toggle_is_resume = False
    with ctrl_cols[2]:
        stop_pressed = st.button(
            "⏹ Stop", disabled=not running, width="stretch",
            help="Sends SIGTERM to the subprocess (like a manual Ctrl-C).",
        )
    with ctrl_cols[3]:
        reset_pressed = st.button(
            "🔄 Reset robot", disabled=running, width="stretch",
            help="Runs `battleship_ot2_loop reset` — homes the arm and drops any attached tip.",
        )

    # ── Button handlers ───────────────────────────────────────────────
    if start_pressed and not running:
        path_arg = (geometry_path or "").strip() or None
        if path_arg and not Path(path_arg).exists():
            st.error(f"Calibration file not found: {path_arg}")
        else:
            cfg = LoopConfig(
                strategy=strategy,
                seed=int(seed),
                robot_ip=robot_ip,
                color_develop_seconds=float(develop_s),
                geometry_path=path_arg,
                output_dir=output_dir,
                dry_run=bool(dry_run),
                skip_setup=bool(skip_setup),
                deck_overrides=dict(st.session_state.get("hw_deck_overrides", {})),
            )
            try:
                ctrl.start(cfg)
                st.rerun()
            except RuntimeError as e:
                st.warning(str(e))
    if toggle_pressed and running:
        if toggle_is_resume:
            ctrl.resume()
        else:
            ctrl.pause()
        st.rerun()
    if stop_pressed and running:
        ctrl.stop()
        st.info(
            "SIGTERM sent. The subprocess will exit immediately; if it's "
            "mid-move the OT-2 may need a **Reset** afterwards."
        )
    if reset_pressed and not running:
        with st.spinner(f"Resetting robot at {robot_ip} …"):
            try:
                result = ctrl.reset_robot(robot_ip)
            except RuntimeError as e:
                st.warning(str(e))
                result = None
        if result is not None:
            if result["returncode"] == 0:
                st.success("Robot reset — arm homed, tip dropped.")
            else:
                st.error(f"Reset exited with rc={result['returncode']}. See output below.")
            with st.expander("Reset command output", expanded=True):
                st.code(ctrl.last_reset_output() or "(no output)", language="text")

    # ── Metrics strip ──────────────────────────────────────────────────
    m = st.columns(5)
    m[0].metric("Step", snap.step)
    m[1].metric("Hits", snap.hits)
    m[2].metric("Misses", snap.misses)
    ships_label = f"{snap.ships_sunk} / {snap.ships_total}" if snap.ships_total else "—"
    m[3].metric("Ships sunk", ships_label)
    remaining = max(0, snap.total_ship_cells - snap.hits)
    m[4].metric("Ship cells remaining", remaining if snap.total_ship_cells else "—")

    # ── Human-check banner (subprocess paused on ambiguous reading) ───
    if snap.human_check_request:
        _render_human_check_panel(ctrl, snap)

    # ── Visualisations ────────────────────────────────────────────────
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Plate — hits / misses**")
        if snap.results_matrix is not None:
            st.pyplot(_plot_results_plate(snap.results_matrix), clear_figure=True)
        else:
            st.caption("Plate view appears after the first query completes.")

        st.markdown("**Belief probability map**")
        if snap.prob_map is not None:
            prob_active = np.asarray(snap.prob_map)[:BOARD_ROWS, :BOARD_COLS]
            # In ship mode prob_map is a placement-density distribution that
            # sums to 1 (so individual cells are ~0.01-0.05) and isn't very
            # readable. Rescale to [0, 1] relative to the current max so
            # the most-likely cell is 1.0 and the colour-bar / annotations
            # are intuitive ("relative belief, normalised to the leader").
            pmax = float(prob_active.max())
            if pmax > 0:
                prob_display = prob_active / pmax
            else:
                prob_display = prob_active
            st.pyplot(
                _plot_board_heatmap(
                    prob_display,
                    title=f"P(ship), normalised (raw max = {pmax:.3f})",
                    cmap="magma", vmin=0.0, vmax=1.0,
                    annotate_values=True,
                ),
                clear_figure=True,
            )
        else:
            st.caption("Belief map appears after the first query completes.")

    with right:
        cam_l, cam_r = st.columns(2)
        with cam_l:
            st.markdown("**Last camera frame**")
            img_path = snap.last_image_path
            if img_path and Path(img_path).exists():
                st.image(img_path, caption=Path(img_path).name, width="stretch")
            else:
                st.caption("No image captured yet.")

        with cam_r:
            live_on = _render_live_preview_panel(running)

        if snap.last_step:
            st.markdown("**Last step**")
            st.json({
                "step": snap.last_step["step"],
                "well": snap.last_step["well_name"],
                "label": snap.last_step["label"],
                "hit": snap.last_step["is_hit"],
                "confidence": round(float(snap.last_step["confidence"]), 3),
                "mean_rgb": snap.last_step["mean_rgb"],
                "sunk_ship_size": snap.last_step.get("sunk_ship_size"),
            })

        if snap.ground_truth is not None:
            with st.expander("Ground truth (seeded board)", expanded=False):
                gt = np.asarray(snap.ground_truth)
                st.pyplot(
                    _plot_board_heatmap(
                        gt, title="ground truth (1 = ship)",
                        cmap="Purples", vmin=0, vmax=1,
                    ),
                    clear_figure=True,
                )

    # ── Manual hole-status correction ─────────────────────────────────
    if snap.history:
        _render_manual_correction_panel(ctrl, snap)

    # ── Step history table ────────────────────────────────────────────
    if snap.history:
        with st.expander(f"Step history ({len(snap.history)})", expanded=False):
            rows = [
                {
                    "step": r["step"],
                    "well": r["well_name"],
                    "label": r["label"],
                    "hit": r["is_hit"],
                    "conf": round(float(r["confidence"]), 3),
                    "sunk": r.get("sunk_ship_size"),
                    "t": r["timestamp"].split("T")[-1][:8],
                }
                for r in snap.history[-50:]
            ]
            st.table(rows)

    # ── Log stream ────────────────────────────────────────────────────
    with st.expander("Live log", expanded=running):
        log_tail = ctrl.logs(tail=200)
        st.code("\n".join(log_tail) if log_tail else "(no output yet)", language="text")

    if snap.output_dir:
        st.caption(f"Output directory: `{snap.output_dir}`")

    # ── Auto-refresh while running OR while live preview is on ────────
    if running or st.session_state.get("hw_live_camera", False):
        time.sleep(1.0)
        st.rerun()


if __name__ == "__main__":
    main()
