"""
Battleship → 96-Well Plate Photo Simulation
============================================
Uses a Battleship ground-truth grid (1 = ship / “hit” liquid, 0 = water) on an
8×10 active experiment area, then embeds it into a full 8×12 plate. The last
two columns are reserved as unused / colour-control wells to match the real
workflow. The final image is a synthetic top-down plate photo similar to
``PlateSimulator`` (lighting, meniscus, reflections, plastic background).

Typical pipeline
----------------
1. ``BattleshipBoard(rows=8, cols=10, seed=...)``  →  active experiment grid
2. ``generate_battleship_plate_image(board.grid, seed=...)``  →  full 8×12 BGR image
3. ``get_fixed_well_geometry()`` matches simulator centres / radius for ROI sampling
4. ``sample_well_mean_bgr(image, row, col, geometry)``  →  mean BGR in inner well disk (fixed ROI)
5. ``query_well_fixed_geometry_rgb(...)``  →  ``"ship"`` / ``"water"`` / ``"unknown"`` via RGB distance to prototypes
"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

import cv2
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plate.battleship_plate_readout import (
    ACTIVE_COLS,
    DEFAULT_RGB_L2_TOLERANCE,
    DEFAULT_RGB_PER_CHANNEL_DELTA,
    SHIP_LIQUID_BGR,
    SHIP_LIQUID_RGB,
    UNUSED_WELL_VALUE,
    WATER_LIQUID_BGR,
    WATER_LIQUID_RGB,
    classify_ship_water_fixed_rgb_tolerance,
    classify_ship_water_from_mean_bgr,
    decode_active_board_from_image_bgr,
    decode_plate_from_image_bgr,
    mean_bgr_to_mean_rgb,
    query_well_fixed_geometry_rgb,
    sample_all_wells_mean_bgr,
    sample_well_mean_bgr,
)
from core.battleship_env import BattleshipBoard
from plate.plate_simulator import PlateSimulator

RESERVED_CONTROL_COLS = 2

__all__ = [
    "ACTIVE_COLS",
    "DEFAULT_RGB_L2_TOLERANCE",
    "DEFAULT_RGB_PER_CHANNEL_DELTA",
    "SHIP_LIQUID_BGR",
    "SHIP_LIQUID_RGB",
    "UNUSED_WELL_VALUE",
    "WATER_LIQUID_BGR",
    "WATER_LIQUID_RGB",
    "board_grid_to_plate_labels",
    "classify_ship_water_fixed_rgb_tolerance",
    "classify_ship_water_from_mean_bgr",
    "decode_active_board_from_image_bgr",
    "decode_plate_from_image_bgr",
    "generate_battleship_plate_image",
    "get_fixed_well_geometry",
    "make_battleship_plate_simulator",
    "mean_bgr_to_mean_rgb",
    "query_well_fixed_geometry_rgb",
    "run_quick_demo",
    "sample_all_wells_mean_bgr",
    "sample_well_mean_bgr",
    "simulate_photo_from_board",
    "to_full_plate_layout",
]


def get_fixed_well_geometry(**sim_kwargs) -> Dict:
    """
    Geometry dict aligned with ``PlateSimulator`` defaults (fixed camera layout).

    Pass through ``sim_kwargs`` to match a customised ``PlateSimulator`` instance,
    e.g. ``well_spacing=80``, ``image_width=1060``. Optional ``seed`` only affects
    the temporary simulator’s RNG (geometry itself is deterministic from layout).
    """
    sim = make_battleship_plate_simulator(**sim_kwargs)
    return sim.get_geometry()


def make_battleship_plate_simulator(
    seed: Optional[int] = None,
    **sim_kwargs,
) -> PlateSimulator:
    """``PlateSimulator`` with red ship liquid and blue water."""
    kw: Dict[str, Any] = {
        "positive_bgr": SHIP_LIQUID_BGR,
        "negative_bgr": WATER_LIQUID_BGR,
    }
    kw.update(sim_kwargs)
    return PlateSimulator(seed=seed, **cast(Dict[str, Any], kw))


def generate_battleship_plate_image(
    ship_mask: np.ndarray,
    seed: Optional[int] = None,
    **sim_kwargs,
) -> np.ndarray:
    """
    Render one synthetic photograph.

    Parameters
    ----------
    ship_mask : (8, 10) or (8, 12) int
        1 = ship (red liquid), 0 = water (blue liquid).
        If ``(8, 10)``, the function automatically pads the last two columns as
        unused / control wells with value ``-1``.
    """
    full_plate = to_full_plate_layout(ship_mask)
    if full_plate.shape != (PlateSimulator.ROWS, PlateSimulator.COLS):
        raise ValueError(
            f"Expected label shape ({PlateSimulator.ROWS}, {PlateSimulator.COLS}), "
            f"got {full_plate.shape}"
        )
    sim = make_battleship_plate_simulator(seed=seed, **sim_kwargs)
    return sim.generate_image(full_plate.astype(int))


def to_full_plate_layout(ship_mask: np.ndarray) -> np.ndarray:
    """
    Convert active-area labels to full 8×12 plate labels.

    The real workflow uses the last two columns as reserved control wells, so
    this function fills them with ``-1``.
    """
    if ship_mask.shape == (PlateSimulator.ROWS, ACTIVE_COLS):
        out = np.full(
            (PlateSimulator.ROWS, PlateSimulator.COLS),
            UNUSED_WELL_VALUE,
            dtype=int,
        )
        out[:, :ACTIVE_COLS] = ship_mask.astype(int)
        return out
    if ship_mask.shape == (PlateSimulator.ROWS, PlateSimulator.COLS):
        out = ship_mask.astype(int).copy()
        out[:, ACTIVE_COLS:] = UNUSED_WELL_VALUE
        return out
    raise ValueError(
        f"Expected active grid (8, 10) or full grid (8, 12), got {ship_mask.shape}"
    )


def board_grid_to_plate_labels(board: BattleshipBoard) -> np.ndarray:
    """Map an 8×10 active Battleship board into the full 8×12 plate layout."""
    g = board.grid
    return to_full_plate_layout(g)


def simulate_photo_from_board(
    board: BattleshipBoard,
    seed: Optional[int] = None,
    **sim_kwargs,
) -> np.ndarray:
    """Convenience: ``generate_battleship_plate_image(board.grid, ...)`` with shape check."""
    return generate_battleship_plate_image(board_grid_to_plate_labels(board), seed=seed, **sim_kwargs)


def run_quick_demo(seed: int = 7, save_path: Optional[str] = None) -> np.ndarray:
    """
    One 8×12 Battleship board → synthetic image → decode; prints simple accuracy.

    Returns the BGR image.
    """
    board = BattleshipBoard(rows=8, cols=ACTIVE_COLS, seed=seed)
    geom = get_fixed_well_geometry(seed=seed)
    img = simulate_photo_from_board(board, seed=seed)
    pred = decode_plate_from_image_bgr(img, geom)
    gt = board_grid_to_plate_labels(board)
    active_mask = gt != UNUSED_WELL_VALUE
    acc = float((pred[active_mask] == gt[active_mask]).mean())
    print(f"[battleship_plate_simulation demo] seed={seed}  well accuracy={acc:.4f}")
    if save_path:
        cv2.imwrite(save_path, img)
        print(f"  saved: {save_path}")
    return img


if __name__ == "__main__":
    run_quick_demo(seed=42)
