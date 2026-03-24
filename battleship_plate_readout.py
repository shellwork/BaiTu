"""
Plate RGB readout utilities.

This module is intentionally standalone: it converts a simulated or real plate
image into a matrix that can be queried like the Battleship oracle state.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# BGR liquids (fixed to match the current synthetic plate configuration)
# Ship RGB(166, 92, 135) -> BGR (135, 92, 166)
# Water RGB(104, 91, 134) -> BGR (134, 91, 104)
SHIP_LIQUID_BGR: np.ndarray = np.array([135, 92, 166], dtype=np.float32)
WATER_LIQUID_BGR: np.ndarray = np.array([134, 91, 104], dtype=np.float32)

# Same prototypes in RGB space for fixed-ROI readout.
SHIP_LIQUID_RGB: np.ndarray = np.array([166.0, 92.0, 135.0], dtype=np.float32)
WATER_LIQUID_RGB: np.ndarray = np.array([104.0, 91.0, 134.0], dtype=np.float32)

DEFAULT_RGB_L2_TOLERANCE: float = 48.0
DEFAULT_RGB_PER_CHANNEL_DELTA: float = 22.0
UNUSED_WELL_VALUE = -1
ACTIVE_COLS = 10


def load_image_bgr(image_path: str) -> np.ndarray:
    """Load a real or synthetic plate photo from disk in OpenCV BGR format."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def mean_bgr_to_mean_rgb(mean_bgr: np.ndarray) -> np.ndarray:
    """OpenCV BGR mean -> RGB vector (float)."""
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
    Compare one well's mean colour to ship/water prototypes in RGB space.

    Returns (label, confidence), where label is "ship" | "water" | "unknown".
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


def _extract_inner_disk_pixels(
    image_bgr: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    fraction: float,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    r_inner = max(1, int(radius * fraction))
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    patch = image_bgr[y0:y1, x0:x1]
    ph, pw = patch.shape[:2]
    yy, xx = np.mgrid[0:ph, 0:pw]
    dist = np.sqrt((xx - (cx - x0)) ** 2 + (yy - (cy - y0)) ** 2)
    return patch[dist < r_inner]


def sample_well_mean_bgr(
    image_bgr: np.ndarray,
    row: int,
    col: int,
    geometry: Dict,
    inner_fraction: float = 0.60,
) -> np.ndarray:
    """Mean BGR inside the inner circular ROI of one well."""
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
    """Return per-well mean BGR with shape (rows, cols, 3)."""
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
    Simple nearest-prototype baseline in BGR space.

    Returns (label, confidence) where label is "ship" or "water".
    """
    m = mean_bgr.astype(np.float32)
    dr = float(np.linalg.norm(m - red_prototype_bgr))
    db = float(np.linalg.norm(m - blue_prototype_bgr))
    if dr + db < 1e-6:
        return "ship", 0.5
    er = np.exp(-dr)
    eb = np.exp(-db)
    s = er + eb
    if dr < db:
        return "ship", float(er / s)
    return "water", float(eb / s)


def query_well_fixed_geometry_rgb(
    image_bgr: np.ndarray,
    row: int,
    col: int,
    geometry: Dict,
    inner_fraction: float = 0.60,
    **classify_kw,
) -> Tuple[str, np.ndarray, float]:
    """
    Read one well and classify it as ship / water / unknown.

    Returns (label, mean_bgr_uint8, confidence).
    """
    mean_b = sample_well_mean_bgr(image_bgr, row, col, geometry, inner_fraction=inner_fraction)
    label, conf = classify_ship_water_fixed_rgb_tolerance(mean_b, **classify_kw)
    return label, mean_b, conf


def decode_plate_from_image_bgr(
    image_bgr: np.ndarray,
    geometry: Dict,
    inner_fraction: float = 0.60,
    active_cols: int = ACTIVE_COLS,
    unknown_fallback: str = "nearest",
    **rgb_classify_kw,
) -> np.ndarray:
    """
    Convert a plate image into a full matrix.

    Output is an int matrix shaped like the plate:
    - active area: 1 = ship, 0 = water
    - reserved columns: -1
    """
    rows, cols = geometry["rows"], geometry["cols"]
    out = np.full((rows, cols), UNUSED_WELL_VALUE, dtype=int)
    for ri in range(rows):
        for ci in range(min(active_cols, cols)):
            label, mean_bgr, _ = query_well_fixed_geometry_rgb(
                image_bgr,
                ri,
                ci,
                geometry,
                inner_fraction=inner_fraction,
                **rgb_classify_kw,
            )
            if label == "ship":
                out[ri, ci] = 1
            elif label == "water":
                out[ri, ci] = 0
            elif unknown_fallback == "nearest":
                mean_rgb = mean_bgr_to_mean_rgb(mean_bgr)
                ds = float(np.linalg.norm(mean_rgb - SHIP_LIQUID_RGB))
                dw = float(np.linalg.norm(mean_rgb - WATER_LIQUID_RGB))
                out[ri, ci] = 1 if ds < dw else 0
            else:
                out[ri, ci] = UNUSED_WELL_VALUE
    return out


def decode_active_board_from_image_bgr(
    image_bgr: np.ndarray,
    geometry: Dict,
    inner_fraction: float = 0.60,
    active_cols: int = ACTIVE_COLS,
    **rgb_classify_kw,
) -> np.ndarray:
    """
    Convert a plate image into the active 8x10 Battleship-style matrix.

    This is the direct drop-in matrix for table lookup during active learning.
    """
    full = decode_plate_from_image_bgr(
        image_bgr,
        geometry,
        inner_fraction=inner_fraction,
        active_cols=active_cols,
        **rgb_classify_kw,
    )
    return full[:, :active_cols].copy()
