"""
Battleship Lab Simulator – Streamlit Dashboard
================================================
Simulates a Battleship game played on an Opentrons OT-2 liquid handler.

Run:
    streamlit run battleship_dashboard.py

Two tabs
--------
1. **Simulation** – runs all 4 strategies in the background, displays the
   hunt-target strategy shot-by-shot on a 96-well plate, and shows a
   comparative hits-vs-round chart for all strategies.
2. **Handle Logical Failures** – same view but unclear CV readings
   (score 0.4–0.6) are paused until the user provides correction input.

The true board layout is always visible at the top.
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

from battleship_env import BattleshipBoard
from battleship_model import Game
from battleship_synthetic import (
    NoiseConfig,
    generate_single_well_reading,
)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

BOARD_ROWS, BOARD_COLS = 8, 10
PLATE_ROWS, PLATE_COLS = 8, 12  # standard 96-well plate

STRATEGIES = ["random", "prob", "entropy", "hunt_target"]
LABELS = {
    "random":      "Random",
    "prob":        "Max Probability",
    "entropy":     "Max Entropy",
    "hunt_target": "Hunt-Target",
}
COLORS = {
    "random":      "#e74c3c",
    "prob":        "#2ecc71",
    "entropy":     "#9b59b6",
    "hunt_target": "#f39c12",
}

# Ship sizes (same as in battleship_env.py)
SHIP_SIZES = [5, 4, 3, 3, 2]
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

def main():
    st.set_page_config(
        page_title="Battleship Lab Simulator",
        page_icon="🧪",
        layout="wide",
    )

    st.title("Battleship Lab Simulator")
    st.markdown(
        """
Simulating a Battleship game on an **Opentrons OT-2** liquid handler.

**Setup:** The OT-2 sets up the board by dispensing NaOH into ship positions and water into empty positions.  
**Play:** In each iteration, cabbage juice is dispensed into the selected well, turning the cells purple if it hits a ship position and blue if it hits an empty cell.  
**Analysis:** A computer vision algorithm takes a picture of the plate and returns an array of values from 0 to 1 where 0 indicates a hit and 1 indicates a miss.  
**Learning:** An active learning algorithm selects the next cell to hit. 4 query selection strategies are employed: max entropy, max probability, hunt-target heuristic, and a random baseline:

---
**How to Play:**  
- Use the strategy buttons at the top to view the progress of each query selection strategy.  
- Click **Next Shot** or **Play 5 Shots** to advance the simulation, or **Play All** to run to completion.  
- The 96-well plate and true board update for the selected strategy.  
- For the **Hunt-Target** strategy, you may be prompted to verify unclear readings. When prompted, select whether the well is a hit (ship) or miss (water) to continue.  
- The dashboard tracks liquid usage, QC metrics, and progress for each strategy independently.  
- You can start a new board at any time with the **New Board** button.

---
""")
    # Reduce padding to fit more on screen
    st.markdown("""
    <style>
        .main { padding-top: 1rem; }
        h2, h3 { margin-top: 0.5rem; margin-bottom: 0.3rem; }
        .element-container { margin-bottom: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

    _render_simulation_tab(prefix="")


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
                st.pyplot(fig_board, use_container_width=True)
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
                if st.button("🎯 HIT (Ship)", key=f"{prefix}dialog_hit_{item_key}", use_container_width=True, type="primary"):
                    # Save response — will be processed on next rerun
                    st.session_state[f"{prefix}dialog_response"] = {
                        "row": item["row"], "col": item["col"], "value": 0
                    }
                    st.rerun()
            with col_miss:
                if st.button("💧 MISS (Water)", key=f"{prefix}dialog_miss_{item_key}", use_container_width=True, type="primary"):
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
            clicked = st.button(btn_label, key=f"{prefix}strat_{strategy}", use_container_width=True, 
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
        st.pyplot(fig_board, use_container_width=True)
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
        st.pyplot(fig_plate, use_container_width=True)
        plt.close(fig_plate)

    # --- Control buttons (under the visuals) ---
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    with col1:
        if st.button("🆕 New Board", key=f"{prefix}btn_new_board", use_container_width=True):
            seed = st.session_state[f"{prefix}next_seed"]
            _reset_session(seed, prefix)
            st.session_state[f"{prefix}next_seed"] = seed + 1
            st.rerun()

    with col2:
        step_disabled = done or paused or auto_play
        if st.button("▶️ Next Shot", key=f"{prefix}btn_step", disabled=step_disabled, use_container_width=True):
            _step_all_strategies(prefix)
            st.rerun()

    with col3:
        if st.button("⏩ Play 5 Shots", key=f"{prefix}btn_auto", disabled=(done or paused or auto_play), use_container_width=True):
            for _ in range(5):
                if not st.session_state[f"{prefix}done"] and len(st.session_state[f"{prefix}unclear_queue"]) == 0:
                    _step_all_strategies(prefix)
                    if len(st.session_state[f"{prefix}unclear_queue"]) > 0:
                        break  # pause on unclear
            st.rerun()

    with col4:
        if auto_play:
            # Show Stop button when auto-playing
            if st.button("⏹️ Stop", key=f"{prefix}btn_stop", use_container_width=True, type="primary"):
                st.session_state[f"{prefix}auto_play"] = False
                st.rerun()
        else:
            if st.button("▶️▶️ Play All", key=f"{prefix}btn_play_all", disabled=(done or paused), use_container_width=True):
                st.session_state[f"{prefix}auto_play"] = True
                st.rerun()

    with col5:
        if st.button("💾 Save", key=f"{prefix}btn_save", use_container_width=True):
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
        st.pyplot(fig_liquid, use_container_width=True)
        plt.close(fig_liquid)
    
    with col_ships:
        total_cells = boards["hunt_target"].total_ship_cells
        fig_hits = _draw_hits_chart(st.session_state[f"{prefix}hits_per_round"], total_cells)
        st.pyplot(fig_hits, use_container_width=True)
        plt.close(fig_hits)

    st.markdown("---")

    # --- ROW 2: QC Metrics (left: Variance | right: Unclear Cells) ---
    st.markdown("### QC Metrics")
    col_variance, col_unclear = st.columns(2)
    
    with col_variance:
        fig_variance = _draw_variance_chart(st.session_state[f"{prefix}cv_readings"])
        st.pyplot(fig_variance, use_container_width=True)
        plt.close(fig_variance)
    
    with col_unclear:
        fig_unclear = _draw_unclear_cells_chart(st.session_state[f"{prefix}unclear_count_per_round"])
        st.pyplot(fig_unclear, use_container_width=True)
        plt.close(fig_unclear)

    st.markdown("---")

    # --- ROW 3: Strategy Summary (full width, always visible) ---
    st.markdown("### Strategy Summary")
    summary_data = []
    for s in STRATEGIES:
        board = boards[s]
        hits_list = st.session_state[f"{prefix}hits_per_round"][s]
        current_hits = hits_list[-1] if hits_list else 0
        summary_data.append({
            "Strategy": LABELS[s],
            "Shots": board.n_queries,
            "Hits": current_hits,
            "Sunk": len(board.get_sunk_ships()),
            "Done": "✅" if board.is_game_over() else "—",
        })
    # Add key based on round number to help Streamlit track table updates
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


if __name__ == "__main__":
    main()
