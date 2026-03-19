"""
96-Well Plate Computer Vision Detector
========================================
Given a plate photograph, detects each well position and classifies
its colour as purple, blue, or unknown.

Two detection modes
-------------------
  grid    : geometry-guided (fast, uses known plate layout)
  hough   : blind Hough-circle detection (robust to unknown geometry)

The output is an 8×12 integer matrix: 1=purple, 0=blue, -1=unknown.
"""

from __future__ import annotations

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple


# ── HSV colour thresholds (OpenCV H: 0-180, S/V: 0-255) ─────────────────
# Purple: standard H ≈ 290° → OpenCV H ≈ 145
# Blue:   standard H ≈ 210° → OpenCV H ≈ 105
PURPLE_H = (128, 162)
BLUE_H   = (88,  128)
MIN_S    = 70
MIN_V    = 55


class PlateDetector:
    """
    CV pipeline: image → 8×12 colour label matrix.

    Parameters
    ----------
    geometry : dict from PlateSimulator.get_geometry()
        If provided, uses fast grid-based detection.
        If None,     falls back to Hough circle detection.
    """

    ROWS = 8
    COLS = 12

    def __init__(self, geometry: Optional[Dict] = None):
        self.geometry = geometry

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Detect all wells and return 8×12 label matrix.

        Labels: 1=purple, 0=blue, -1=unknown
        """
        if self.geometry is not None:
            return self._process_grid(image)

        labels, _ = self._process_hough(image)
        return labels

    def query_well(
        self, image: np.ndarray, row: int, col: int
    ) -> Tuple[str, np.ndarray, float]:
        """
        Classify a single well by position.  Used during active learning
        to simulate reading one well without processing the full plate.

        Returns
        -------
        label    : 'purple' | 'blue' | 'unknown'
        mean_bgr : np.ndarray (3,)
        confidence : float [0, 1]
        """
        assert self.geometry is not None, \
            "query_well requires geometry; use PlateSimulator.get_geometry()"

        g = self.geometry
        cx = g["col_centers"][col]
        cy = g["row_centers"][row]
        r  = g["well_radius"]
        return self._classify_well(image, cx, cy, r)

    # ------------------------------------------------------------------
    # Grid-based detection (fast)
    # ------------------------------------------------------------------

    def _process_grid(self, image: np.ndarray) -> np.ndarray:
        g = self.geometry
        labels = np.full((g["rows"], g["cols"]), -1, dtype=int)

        for ri, cy in enumerate(g["row_centers"]):
            for ci, cx in enumerate(g["col_centers"]):
                label, _, _ = self._classify_well(image, cx, cy, g["well_radius"])
                labels[ri, ci] = {"purple": 1, "blue": 0}.get(label, -1)

        return labels

    # ------------------------------------------------------------------
    # Hough-circle detection (blind)
    # ------------------------------------------------------------------

    def _process_hough(
        self, image: np.ndarray, expected_radius: int = 28
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Detect circles, snap to grid, then classify."""
        circles = self._hough_circles(image, expected_radius)
        empty   = np.full((self.ROWS, self.COLS), -1, dtype=int)

        if circles is None or len(circles) < 0.5 * self.ROWS * self.COLS:
            return empty, None

        row_c = self._cluster_1d(circles[:, 1], self.ROWS)
        col_c = self._cluster_1d(circles[:, 0], self.COLS)

        if row_c is None or col_c is None:
            return empty, None

        median_r = int(np.median(circles[:, 2]))
        labels   = np.full((self.ROWS, self.COLS), -1, dtype=int)

        for ri, cy in enumerate(sorted(row_c)):
            for ci, cx in enumerate(sorted(col_c)):
                label, _, _ = self._classify_well(image, cx, cy, median_r)
                labels[ri, ci] = {"purple": 1, "blue": 0}.get(label, -1)

        detected_geom = {
            "rows":        self.ROWS,
            "cols":        self.COLS,
            "row_centers": sorted(row_c),
            "col_centers": sorted(col_c),
            "well_radius": median_r,
        }
        return labels, detected_geom

    def _hough_circles(
        self, image: np.ndarray, expected_radius: int
    ) -> Optional[np.ndarray]:
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), sigmaX=2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp        = 1,
            minDist   = int(expected_radius * 1.8),
            param1    = 50,
            param2    = 28,
            minRadius = int(expected_radius * 0.65),
            maxRadius = int(expected_radius * 1.45),
        )
        return circles[0] if circles is not None else None

    @staticmethod
    def _cluster_1d(values: np.ndarray, k: int) -> Optional[List[int]]:
        """
        Collapse detected circle coordinates into exactly k cluster centres
        via repeated nearest-pair merging.
        """
        centers = sorted(set(np.round(values).astype(int).tolist()))
        while len(centers) > k:
            diffs  = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
            i_min  = int(np.argmin(diffs))
            merged = (centers[i_min] + centers[i_min + 1]) // 2
            centers = centers[:i_min] + [merged] + centers[i_min + 2:]
        return centers if len(centers) == k else None

    # ------------------------------------------------------------------
    # Colour classification
    # ------------------------------------------------------------------

    def _classify_well(
        self, image: np.ndarray, cx: int, cy: int, radius: int
    ) -> Tuple[str, np.ndarray, float]:
        """
        Classify a single well at pixel position (cx, cy).

        Extracts pixels from the inner 60% of the well radius to avoid
        contamination from the wall ring, then classifies hue in HSV.

        Returns (label, mean_bgr, confidence).
        """
        pixels = self._extract_inner_pixels(image, cx, cy, radius, fraction=0.60)

        if len(pixels) == 0:
            return "unknown", np.zeros(3, dtype=np.uint8), 0.0

        mean_bgr = pixels.mean(axis=0).astype(np.uint8)

        # Convert sample pixels to HSV for robust hue classification
        px_bgr = pixels.astype(np.uint8).reshape(-1, 1, 3)
        px_hsv = cv2.cvtColor(px_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)

        mean_H = float(px_hsv[:, 0].mean())
        mean_S = float(px_hsv[:, 1].mean())
        mean_V = float(px_hsv[:, 2].mean())

        # Fraction of pixels that land in each colour range
        purple_mask = (
            (px_hsv[:, 0] >= PURPLE_H[0]) & (px_hsv[:, 0] <= PURPLE_H[1]) &
            (px_hsv[:, 1] >= MIN_S) & (px_hsv[:, 2] >= MIN_V)
        )
        blue_mask = (
            (px_hsv[:, 0] >= BLUE_H[0]) & (px_hsv[:, 0] <= BLUE_H[1]) &
            (px_hsv[:, 1] >= MIN_S) & (px_hsv[:, 2] >= MIN_V)
        )

        n = len(px_hsv)
        frac_purple = purple_mask.sum() / n
        frac_blue   = blue_mask.sum()   / n

        if frac_purple > frac_blue and frac_purple > 0.35:
            return "purple", mean_bgr, float(frac_purple)
        if frac_blue > frac_purple and frac_blue > 0.35:
            return "blue", mean_bgr, float(frac_blue)

        # Tie-break using mean hue
        if mean_S >= MIN_S and mean_V >= MIN_V:
            if PURPLE_H[0] <= mean_H <= PURPLE_H[1]:
                return "purple", mean_bgr, 0.5
            if BLUE_H[0] <= mean_H <= BLUE_H[1]:
                return "blue", mean_bgr, 0.5

        return "unknown", mean_bgr, 0.0

    @staticmethod
    def _extract_inner_pixels(
        image: np.ndarray,
        cx: int, cy: int,
        radius: int,
        fraction: float = 0.60,
    ) -> np.ndarray:
        """Return BGR pixels inside the inner circle of a well."""
        H, W = image.shape[:2]
        r_inner = max(1, int(radius * fraction))

        y0, y1 = max(0, cy - radius), min(H, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(W, cx + radius + 1)
        patch = image[y0:y1, x0:x1]

        ph, pw = patch.shape[:2]
        yy, xx = np.mgrid[0:ph, 0:pw]
        dist   = np.sqrt((xx - (cx - x0)) ** 2 + (yy - (cy - y0)) ** 2)

        return patch[dist < r_inner]          # shape (N, 3)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualise(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        title: str = "Detected Wells",
        ax=None,
    ):
        """
        Draw detected well circles colour-coded by classification result.
        Green = purple, Cyan = blue, Red = unknown.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        g   = self.geometry or {}

        if ax is None:
            _, ax = plt.subplots(figsize=(14, 9))

        ax.imshow(rgb)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

        COLOUR_MAP = {1: "#cc44ee", 0: "#3399ff", -1: "#ff4444"}
        LABEL_MAP  = {1: "purple", 0: "blue",  -1: "unknown"}

        if g and "row_centers" in g:
            for ri, cy in enumerate(g["row_centers"]):
                for ci, cx in enumerate(g["col_centers"]):
                    lbl  = int(labels[ri, ci])
                    col  = COLOUR_MAP[lbl]
                    circ = plt.Circle(
                        (cx, cy), g["well_radius"] + 3,
                        fill=False, edgecolor=col, linewidth=2.0
                    )
                    ax.add_patch(circ)

        patches = [
            mpatches.Patch(color=COLOUR_MAP[k], label=LABEL_MAP[k])
            for k in [1, 0, -1]
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=8)
        return ax

    def accuracy(self, detected: np.ndarray, ground_truth: np.ndarray) -> Dict:
        """Compare detected labels to ground truth; return accuracy metrics."""
        known   = detected != -1
        correct = (detected[known] == ground_truth[known]).sum()
        tp = ((detected == 1) & (ground_truth == 1)).sum()
        fp = ((detected == 1) & (ground_truth == 0)).sum()
        fn = ((detected == 0) & (ground_truth == 1)).sum()

        return {
            "n_wells":     int(detected.size),
            "n_detected":  int(known.sum()),
            "accuracy":    float(correct / known.sum()) if known.any() else 0.0,
            "precision":   float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "recall":      float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "n_unknown":   int((detected == -1).sum()),
        }
