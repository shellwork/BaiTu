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

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from battleship_env import BattleshipBoard
from plate_simulator import PlateSimulator

# BGR liquids (fixed to match user’s real plate photo; RGB given as ship / water)
# Ship RGB(166, 92, 135) → BGR (135, 92, 166)
# Water RGB(104, 91, 134) → BGR (134, 91, 104)
SHIP_LIQUID_BGR: np.ndarray = np.array([135, 92, 166], dtype=np.float32)
WATER_LIQUID_BGR: np.ndarray = np.array([134, 91, 104], dtype=np.float32)
# Same prototypes in RGB space (for fixed-ROI RGB distance readout)
SHIP_LIQUID_RGB: np.ndarray = np.array([166.0, 92.0, 135.0], dtype=np.float32)
WATER_LIQUID_RGB: np.ndarray = np.array([104.0, 91.0, 134.0], dtype=np.float32)
# Default tolerances: geometry is fixed; liquid colour varies with simulator noise.
# Tune with calibration on your rig; L2 ball ~95% of synthetic wells at default render settings.
DEFAULT_RGB_L2_TOLERANCE: float = 48.0
DEFAULT_RGB_PER_CHANNEL_DELTA: float = 22.0
UNUSED_WELL_VALUE = -1
ACTIVE_COLS = 10
RESERVED_CONTROL_COLS = 2


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
    kw = dict(positive_bgr=SHIP_LIQUID_BGR, negative_bgr=WATER_LIQUID_BGR)
    kw.update(sim_kwargs)
    return PlateSimulator(seed=seed, **kw)


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


def mean_bgr_to_mean_rgb(mean_bgr: np.ndarray) -> np.ndarray:
    """OpenCV BGR mean → RGB vector (float)."""
    b, g, r = mean_bgr.astype(np.float32).reshape(3)
    return np.array([r, g, b], dtype=np.float32)


def classify_ship_water_fixed_rgb_tolerance(
    mean_bgr: np.ndarray,
    ship_rgb: np.ndarray = SHIP_LIQUID_RGB,
    water_rgb: np.ndarray = WATER_LIQUID_RGB,
    *,
    l2_max: float = DEFAULT_RGB_L2_TOLERANCE,
    per_channel_delta: Optional[float] = None,
) -> Tuple[str, float]:
    """
    Fixed-geometry readout: compare mean ROI colour to predefined ship/water RGB.

    - If ``per_channel_delta`` is None: accept class if Euclidean distance in RGB
      is <= ``l2_max`` (ball tolerance, "CI" as radius in colour space).
    - If ``per_channel_delta`` is set: additionally require each channel within
      ±delta of the prototype (box ∩ ball, stricter).

    Returns (label, confidence) with label ``ship`` | ``water`` | ``unknown``.
    """
    m = mean_bgr_to_mean_rgb(mean_bgr)
    ds = float(np.linalg.norm(m - ship_rgb))
    dw = float(np.linalg.norm(m - water_rgb))

    def in_ball_ship() -> bool:
        return ds <= l2_max

    def in_ball_water() -> bool:
        return dw <= l2_max

    def in_box(vec: np.ndarray, proto: np.ndarray) -> bool:
        if per_channel_delta is None:
            return True
        return bool(np.all(np.abs(vec - proto) <= per_channel_delta))

    in_s = in_ball_ship() and in_box(m, ship_rgb)
    in_w = in_ball_water() and in_box(m, water_rgb)

    if in_s and in_w:
        if ds <= dw:
            conf = max(0.0, min(1.0, 1.0 - ds / max(l2_max, 1e-6)))
            return "ship", conf
        conf = max(0.0, min(1.0, 1.0 - dw / max(l2_max, 1e-6)))
        return "water", conf
    if in_s:
        return "ship", max(0.0, min(1.0, 1.0 - ds / max(l2_max, 1e-6)))
    if in_w:
        return "water", max(0.0, min(1.0, 1.0 - dw / max(l2_max, 1e-6)))
    return "unknown", 0.0


def query_well_fixed_geometry_rgb(
    image_bgr: np.ndarray,
    row: int,
    col: int,
    geometry: Dict,
    inner_fraction: float = 0.60,
    **classify_kw,
) -> Tuple[str, np.ndarray, float]:
    """
    Sample the same inner-disk ROI as rendering geometry, classify by RGB tolerance.

    Returns (label, mean_bgr_uint8, confidence) with label ``ship`` | ``water`` | ``unknown``.
    """
    mean_b = sample_well_mean_bgr(image_bgr, row, col, geometry, inner_fraction=inner_fraction)
    label, conf = classify_ship_water_fixed_rgb_tolerance(mean_b, **classify_kw)
    return label, mean_b, conf


def sample_well_mean_bgr(
    image_bgr: np.ndarray,
    row: int,
    col: int,
    geometry: Dict,
    inner_fraction: float = 0.60,
) -> np.ndarray:
    """
    Mean BGR inside the inner circular ROI of one well (avoids most of the wall ring).

    This mirrors fixed pixel sampling in a real rig: ``geometry`` supplies the same
    ``row_centers`` / ``col_centers`` / ``well_radius`` used when rendering.
    """
    cx = int(geometry["col_centers"][col])
    cy = int(geometry["row_centers"][row])
    r = int(geometry["well_radius"])
    pixels = _extract_inner_disk_pixels(image_bgr, cx, cy, r, inner_fraction)
    if len(pixels) == 0:
        return np.zeros(3, dtype=np.uint8)
    return pixels.mean(axis=0).astype(np.uint8)


def sample_all_wells_mean_bgr(
    image_bgr: np.ndarray,
    geometry: Dict,
    inner_fraction: float = 0.60,
) -> np.ndarray:
    """Shape (8, 12, 3) float32 mean BGR per well."""
    rows, cols = geometry["rows"], geometry["cols"]
    out = np.zeros((rows, cols, 3), dtype=np.float32)
    for ri in range(rows):
        for ci in range(cols):
            out[ri, ci] = sample_well_mean_bgr(
                image_bgr, ri, ci, geometry, inner_fraction=inner_fraction
            )
    return out


def classify_ship_water_from_mean_bgr(
    mean_bgr: np.ndarray,
    red_prototype_bgr: np.ndarray = SHIP_LIQUID_BGR,
    blue_prototype_bgr: np.ndarray = WATER_LIQUID_BGR,
) -> Tuple[str, float]:
    """
    Nearest prototype in BGR (simple baseline for synthetic images).

    Returns (label, confidence) where label is ``"ship"`` or ``"water"`` and
    confidence is normalised membership in [0.5, 1] from softmax over negative distances.
    """
    m = mean_bgr.astype(np.float32)
    dr = float(np.linalg.norm(m - red_prototype_bgr))
    db = float(np.linalg.norm(m - blue_prototype_bgr))
    if dr + db < 1e-6:
        return "ship", 0.5
    # softmax on -distance (closer → higher weight)
    er = np.exp(-dr)
    eb = np.exp(-db)
    s = er + eb
    if dr < db:
        return "ship", float(er / s)
    return "water", float(eb / s)


def decode_plate_from_image_bgr(
    image_bgr: np.ndarray,
    geometry: Dict,
    inner_fraction: float = 0.60,
    active_cols: int = ACTIVE_COLS,
    **rgb_classify_kw,
) -> np.ndarray:
    """
    Build an 8×12 matrix: active area is 1 = ship (red), 0 = water (blue),
    reserved control columns are fixed to ``-1``.

    Uses fixed geometry ROI + RGB tolerance (same as ``query_well_fixed_geometry_rgb``).
    Optional kwargs: ``l2_max``, ``per_channel_delta``, ``ship_rgb``, ``water_rgb``.
    """
    rows, cols = geometry["rows"], geometry["cols"]
    out = np.full((rows, cols), UNUSED_WELL_VALUE, dtype=int)
    for ri in range(rows):
        for ci in range(min(active_cols, cols)):
            label, _, _ = query_well_fixed_geometry_rgb(
                image_bgr, ri, ci, geometry,
                inner_fraction=inner_fraction,
                **rgb_classify_kw,
            )
            if label == "ship":
                out[ri, ci] = 1
            elif label == "water":
                out[ri, ci] = 0
            else:
                m = mean_bgr_to_mean_rgb(mean_b)
                ds = float(np.linalg.norm(m - SHIP_LIQUID_RGB))
                dw = float(np.linalg.norm(m - WATER_LIQUID_RGB))
                out[ri, ci] = 1 if ds < dw else 0
    return out


def _extract_inner_disk_pixels(
    image: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    fraction: float,
) -> np.ndarray:
    H, W = image.shape[:2]
    r_inner = max(1, int(radius * fraction))
    y0, y1 = max(0, cy - radius), min(H, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(W, cx + radius + 1)
    patch = image[y0:y1, x0:x1]
    ph, pw = patch.shape[:2]
    yy, xx = np.mgrid[0:ph, 0:pw]
    dist = np.sqrt((xx - (cx - x0)) ** 2 + (yy - (cy - y0)) ** 2)
    return patch[dist < r_inner]


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
