"""
Plate well detection via adaptive Hough circle detection.

Extracted from BaituOT2Battleship/plateAnalysis.py as a clean, importable module.
Provides functions to detect 96-well plate positions from camera images and
extract per-well RGB values.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

ROWS, COLS = 8, 12


def cluster_1d(values: np.ndarray, k: int) -> Optional[List[int]]:
    """Collapse detected circle coordinates into exactly *k* cluster centres."""
    centers = sorted(set(np.round(values).astype(int).tolist()))
    while len(centers) > k:
        diffs = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        i_min = int(np.argmin(diffs))
        merged = (centers[i_min] + centers[i_min + 1]) // 2
        centers = centers[:i_min] + [merged] + centers[i_min + 2:]
    return centers if len(centers) == k else None


def hough_circles_adaptive(
    image: np.ndarray,
    expected_radius: int,
) -> Optional[np.ndarray]:
    """Hough circle detection with a given expected radius."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), sigmaX=2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(expected_radius * 1.8),
        param1=50,
        param2=28,
        minRadius=int(expected_radius * 0.65),
        maxRadius=int(expected_radius * 1.45),
    )
    return circles[0] if circles is not None else None


def detect_wells_adaptive(image_path: str | Path) -> Optional[Dict]:
    """
    Detect 96-well plate wells using adaptive Hough circle detection.

    Tries multiple expected radii and picks the result closest to 96 circles,
    then clusters into an 8x12 grid and extracts per-well RGB.

    Returns
    -------
    dict with keys:
        wells          : list[dict] – 96 dicts with row, col, center, radius, rgb
        rgb_matrix     : np.ndarray (8, 12) of object – each cell is an (R, G, B) tuple
        circles        : np.ndarray – raw Hough circles
        row_centers    : list[int]  – 8 sorted Y-coordinates
        col_centers    : list[int]  – 12 sorted X-coordinates
        median_radius  : int        – median detected circle radius in pixels
        image          : np.ndarray – original BGR image
    or None if detection fails.
    """
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        log.error("Failed to read image: %s", image_path)
        return None

    log.info("Image loaded: %s  shape=%s", image_path.name, img.shape)

    radius_attempts = [28, 35, 42, 50, 22, 18, 60, 75]
    best_circles: Optional[np.ndarray] = None
    best_count = 0
    best_radius = 28

    for expected_radius in radius_attempts:
        circles = hough_circles_adaptive(img, expected_radius)
        if circles is not None:
            n_circles = len(circles)
            log.debug("  radius %d: %d circles", expected_radius, n_circles)
            if abs(n_circles - 96) < abs(best_count - 96):
                best_circles = circles
                best_count = n_circles
                best_radius = expected_radius

    if best_circles is None or best_count < 48:
        log.warning(
            "Not enough circles detected (got %d, need >= 48). "
            "Manual annotation may be needed.",
            best_count,
        )
        return None

    log.info("Best: %d circles with expected_radius=%d", best_count, best_radius)

    row_centers = cluster_1d(best_circles[:, 1], ROWS)
    col_centers = cluster_1d(best_circles[:, 0], COLS)

    if row_centers is None or col_centers is None:
        log.warning("Could not cluster circles into 8x12 grid.")
        return None

    median_radius = int(np.median(best_circles[:, 2]))
    log.info("Clustered into 8x12 grid, median radius: %d px", median_radius)

    wells_data: List[Dict] = []
    rgb_matrix = np.zeros((ROWS, COLS), dtype=object)
    H, W = img.shape[:2]

    for ri, cy in enumerate(sorted(row_centers)):
        for ci, cx in enumerate(sorted(col_centers)):
            r_inner = max(1, int(median_radius * 0.60))
            y0, y1 = max(0, cy - median_radius), min(H, cy + median_radius + 1)
            x0, x1 = max(0, cx - median_radius), min(W, cx + median_radius + 1)
            patch = img[y0:y1, x0:x1]

            if patch.size > 0:
                ph, pw = patch.shape[:2]
                yy, xx = np.mgrid[0:ph, 0:pw]
                dist = np.sqrt((xx - (cx - x0)) ** 2 + (yy - (cy - y0)) ** 2)
                inner_pixels = patch[dist < r_inner]

                if len(inner_pixels) > 0:
                    b_mean = int(np.mean(inner_pixels[:, 0]))
                    g_mean = int(np.mean(inner_pixels[:, 1]))
                    r_mean = int(np.mean(inner_pixels[:, 2]))
                    rgb = (r_mean, g_mean, b_mean)
                else:
                    rgb = (0, 0, 0)
            else:
                rgb = (0, 0, 0)

            wells_data.append({
                "row": ri,
                "col": ci,
                "center": (int(cx), int(cy)),
                "radius": median_radius,
                "rgb": rgb,
            })
            rgb_matrix[ri, ci] = rgb

    log.info("Extracted RGB from %d wells", len(wells_data))

    return {
        "wells": wells_data,
        "rgb_matrix": rgb_matrix,
        "circles": best_circles,
        "row_centers": sorted(row_centers),
        "col_centers": sorted(col_centers),
        "median_radius": median_radius,
        "image": img,
    }


def build_geometry_from_detection(detect_result: Dict) -> Dict:
    """
    Convert detect_wells_adaptive() output into the geometry dict
    expected by battleship_plate_readout.py functions.
    """
    return {
        "rows": 8,
        "cols": 12,
        "row_centers": detect_result["row_centers"],
        "col_centers": detect_result["col_centers"],
        "well_radius": detect_result["median_radius"],
    }
