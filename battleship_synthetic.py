"""
Battleship Synthetic Lab Data Generator
========================================
Simulates a computer-vision pipeline that reads cabbage-juice indicator
colour from an Opentrons OT-2 96-well plate.

Physical setup
--------------
- All wells start the same colour (clear / pale).
- Ships are pre-loaded with NaOH; empty wells contain water.
- Each "shot" pipettes cabbage juice into the target well.
  - NaOH + cabbage juice  → colour change  (hit)
  - Water + cabbage juice → no change       (miss)

CV pipeline output
------------------
A 10×8 float array in [0, 1]:
  0.0 → confident HIT   (colour change detected)
  1.0 → confident MISS  (no colour change)

Gaussian noise is added to simulate camera / lighting variation and
occasional QC failures. Wells with scores in [0.4, 0.6] are flagged
as *unclear* and require human review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------
# Configurable noise model
# ------------------------------------------------------------------

@dataclass
class NoiseConfig:
    """Parameters for the synthetic CV noise model."""
    hit_mean: float = 0.05       # CV score centre for a HIT
    miss_mean: float = 0.95      # CV score centre for a MISS
    hit_std: float = 0.12        # spread for HIT readings
    miss_std: float = 0.12       # spread for MISS readings
    qc_failure_rate: float = 0.03  # fraction of wells that get a large random shift
    qc_shift_std: float = 0.35   # std of the extra shift on QC-failure wells
    unclear_low: float = 0.4     # lower bound for "unclear" zone
    unclear_high: float = 0.6    # upper bound for "unclear" zone
    seed: Optional[int] = None


# ------------------------------------------------------------------
# Single-well reading
# ------------------------------------------------------------------

def _generate_cv_reading(
    is_hit: bool,
    cfg: NoiseConfig,
    rng: np.random.Generator,
) -> float:
    """
    Return a synthetic CV score for one well.

    Parameters
    ----------
    is_hit : bool
        True if the well contains NaOH (ship present).
    cfg : NoiseConfig
        Noise model parameters.
    rng : numpy Generator
        Random number generator for reproducibility.

    Returns
    -------
    score : float in [0, 1]
    """
    if is_hit:
        score = rng.normal(cfg.hit_mean, cfg.hit_std)
    else:
        score = rng.normal(cfg.miss_mean, cfg.miss_std)

    # occasional QC failure (e.g. bubble, lighting glare)
    if rng.random() < cfg.qc_failure_rate:
        score += rng.normal(0, cfg.qc_shift_std)

    return float(np.clip(score, 0.0, 1.0))


# ------------------------------------------------------------------
# Full-board synthetic reading
# ------------------------------------------------------------------

@dataclass
class CVReading:
    """Result of one synthetic CV frame for the whole board."""
    scores: np.ndarray           # (rows, cols) float in [0, 1]
    is_unclear: np.ndarray       # (rows, cols) bool – True where score ∈ [0.4, 0.6]
    binary_calls: np.ndarray     # (rows, cols) int – 0=hit, 1=miss, -1=unclear
    queried_mask: np.ndarray     # (rows, cols) bool – True for wells that have been shot


def generate_board_reading(
    true_grid: np.ndarray,
    queried_mask: np.ndarray,
    cfg: Optional[NoiseConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> CVReading:
    """
    Generate synthetic CV readings for all *queried* wells on the board.

    Parameters
    ----------
    true_grid : ndarray (rows, cols) of {0, 1}
        Ground-truth board. 1 = ship (NaOH), 0 = water.
    queried_mask : ndarray (rows, cols) of bool
        True for wells that have been shot so far.
    cfg : NoiseConfig, optional
    rng : numpy Generator, optional

    Returns
    -------
    CVReading with per-well scores, flags, and binary calls.
    """
    if cfg is None:
        cfg = NoiseConfig()
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    rows, cols = true_grid.shape
    scores = np.full((rows, cols), np.nan)
    is_unclear = np.zeros((rows, cols), dtype=bool)
    binary_calls = np.full((rows, cols), -2, dtype=int)  # -2 = not yet queried

    for r in range(rows):
        for c in range(cols):
            if not queried_mask[r, c]:
                continue
            is_hit = bool(true_grid[r, c])
            score = _generate_cv_reading(is_hit, cfg, rng)
            scores[r, c] = score

            if cfg.unclear_low <= score <= cfg.unclear_high:
                is_unclear[r, c] = True
                binary_calls[r, c] = -1          # unclear
            elif score < 0.5:
                binary_calls[r, c] = 0           # hit
            else:
                binary_calls[r, c] = 1           # miss

    return CVReading(
        scores=scores,
        is_unclear=is_unclear,
        binary_calls=binary_calls,
        queried_mask=queried_mask.copy(),
    )


# ------------------------------------------------------------------
# Convenience: generate a single-well reading for step-by-step play
# ------------------------------------------------------------------

def generate_single_well_reading(
    is_hit: bool,
    cfg: Optional[NoiseConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, bool, int]:
    """
    Generate a single CV reading for one newly-queried well.

    Returns
    -------
    score     : float in [0, 1]
    unclear   : bool
    binary    : int  (0 = hit, 1 = miss, -1 = unclear)
    """
    if cfg is None:
        cfg = NoiseConfig()
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    score = _generate_cv_reading(is_hit, cfg, rng)
    unclear = cfg.unclear_low <= score <= cfg.unclear_high
    if unclear:
        binary = -1
    elif score < 0.5:
        binary = 0   # hit
    else:
        binary = 1   # miss
    return score, unclear, binary
