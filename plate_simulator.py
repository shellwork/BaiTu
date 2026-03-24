"""
96-Well Plate Image Simulator
==============================
Generates realistic synthetic photographs of a 96-well plate where each
well contains either a purple (positive) or blue (negative) reagent.

Noise sources modelled
----------------------
  - Per-well colour variation  (reagent concentration differences)
  - Meniscus brightness gradient inside each well
  - Reflection highlight       (specular reflection on liquid surface)
  - Global illumination gradient (non-uniform lighting)
  - Background texture          (plastic surface roughness)
  - Shot noise                  (camera sensor noise)
  - Slight motion/focus blur

All images are BGR uint8, compatible with OpenCV.
"""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, Tuple


class PlateSimulator:
    """
    Simulate a 96-well (8×12) plate photograph.

    Active Learning Analogy
    -----------------------
      purple well  ↔  ship cell   (label = 1)
      blue well    ↔  water cell  (label = 0)
      photograph   ↔  pool of unlabelled data
    """

    ROWS: int = 8    # A-H
    COLS: int = 12   # 1-12

    # ── Colour palette (BGR) ──────────────────────────────────────────
    # Purple reagent:  RGB (155, 35, 170)  →  BGR (170, 35, 155)
    # Blue reagent:    RGB (40, 120, 200)  →  BGR (200, 120, 40)
    # Background:      steel workbench     →  BGR (174, 178, 176)
    # Well wall ring:  clear plastic ring  →  BGR (220, 222, 224)
    PURPLE_BGR: np.ndarray = np.array([170, 35, 155], dtype=np.float32)
    BLUE_BGR:   np.ndarray = np.array([200, 120,  40], dtype=np.float32)
    BG_BGR:     np.ndarray = np.array([174, 178, 176], dtype=np.float32)
    WALL_BGR:   np.ndarray = np.array([220, 222, 224], dtype=np.float32)
    EMPTY_BGR:  np.ndarray = np.array([198, 201, 205], dtype=np.float32)
    FRAME_BGR:  np.ndarray = np.array([208, 212, 214], dtype=np.float32)

    def __init__(
        self,
        image_width:  int   = 1060,
        image_height: int   = 740,
        well_spacing: int   = 80,
        well_radius:  int   = 28,
        noise_std:    float = 10.0,
        seed:         Optional[int] = None,
        positive_bgr: Optional[np.ndarray] = None,
        negative_bgr: Optional[np.ndarray] = None,
        empty_bgr:    Optional[np.ndarray] = None,
    ):
        self.W            = image_width
        self.H            = image_height
        self.well_spacing = well_spacing
        self.well_radius  = well_radius
        self.noise_std    = noise_std
        self.rng          = np.random.RandomState(seed)
        # Liquid colours: default purple/blue (screening); override e.g. red/blue for ship/water
        self._positive_bgr = (
            np.asarray(positive_bgr, dtype=np.float32)
            if positive_bgr is not None
            else self.PURPLE_BGR.copy()
        )
        self._negative_bgr = (
            np.asarray(negative_bgr, dtype=np.float32)
            if negative_bgr is not None
            else self.BLUE_BGR.copy()
        )
        self._empty_bgr = (
            np.asarray(empty_bgr, dtype=np.float32)
            if empty_bgr is not None
            else self.EMPTY_BGR.copy()
        )

        # Compute well-centre positions (centred on the image)
        x0 = (self.W - well_spacing * (self.COLS - 1)) // 2
        y0 = (self.H - well_spacing * (self.ROWS - 1)) // 2
        self.col_centers = [x0 + i * well_spacing for i in range(self.COLS)]
        self.row_centers = [y0 + i * well_spacing for i in range(self.ROWS)]

        # Pixel coordinate grids (pre-computed once)
        self._xx, self._yy = np.meshgrid(
            np.arange(self.W), np.arange(self.H), indexing="xy"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_image(self, well_labels: np.ndarray) -> np.ndarray:
        """
        Generate a realistic plate photograph.

        Parameters
        ----------
        well_labels : shape (8, 12) int
            1 = positive / ship colour
            0 = negative / water colour
           -1 = unused or colour-control well (clear / transparent)

        Returns
        -------
        image : shape (H, W, 3) uint8, BGR
        """
        assert well_labels.shape == (self.ROWS, self.COLS), \
            f"Expected (8, 12), got {well_labels.shape}"

        img = self._make_background()
        self._draw_plate_frame(img)
        self._apply_illumination_gradient(img)

        for row_i in range(self.ROWS):
            for col_i in range(self.COLS):
                cx = self.col_centers[col_i]
                cy = self.row_centers[row_i]
                state = int(well_labels[row_i, col_i])
                self._draw_well(img, cx, cy, state)

        self._add_global_noise(img)

        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        img_u8 = cv2.GaussianBlur(img_u8, (3, 3), sigmaX=0.8)
        return img_u8

    def get_geometry(self) -> Dict:
        """Return plate geometry dict consumed by PlateDetector."""
        return {
            "rows":        self.ROWS,
            "cols":        self.COLS,
            "row_centers": list(self.row_centers),
            "col_centers": list(self.col_centers),
            "well_radius": self.well_radius,
        }

    def get_well_center(self, row: int, col: int) -> Tuple[int, int]:
        return (self.col_centers[col], self.row_centers[row])

    # ------------------------------------------------------------------
    # Internal rendering helpers
    # ------------------------------------------------------------------

    def _make_background(self) -> np.ndarray:
        img = np.tile(self.BG_BGR, (self.H, self.W, 1)).astype(np.float32)
        # Stainless-bench style texture
        texture = self.rng.normal(0, 4, img.shape).astype(np.float32)
        img += texture
        img += self.rng.normal(0, 2, (self.H, self.W, 1)).astype(np.float32)
        # Slightly brighter plate area
        r = self.well_radius
        x0 = self.col_centers[0]  - r - 35
        x1 = self.col_centers[-1] + r + 35
        y0 = self.row_centers[0]  - r - 35
        y1 = self.row_centers[-1] + r + 35
        img[y0:y1, x0:x1] += 10.0
        return img

    def _draw_plate_frame(self, img: np.ndarray):
        """Add a translucent rectangular plate body closer to the reference photo."""
        r = self.well_radius
        x0 = self.col_centers[0]  - r - 38
        x1 = self.col_centers[-1] + r + 38
        y0 = self.row_centers[0]  - r - 38
        y1 = self.row_centers[-1] + r + 38

        frame = img[y0:y1, x0:x1]
        frame *= 0.52
        frame += self.FRAME_BGR * 0.48

        # Brighter upper-left edge, darker lower-right edge.
        img[y0:y0 + 6, x0:x1] += 16.0
        img[y0:y1, x0:x0 + 6] += 12.0
        img[y1 - 6:y1, x0:x1] -= 12.0
        img[y0:y1, x1 - 6:x1] -= 8.0

    def _apply_illumination_gradient(self, img: np.ndarray):
        """Simulate non-uniform illumination closer to phone camera capture."""
        cx, cy = self.W * 0.46, self.H * 0.40
        dist = np.sqrt((self._xx - cx) ** 2 + (self._yy - cy) ** 2)
        max_d = np.sqrt(cx ** 2 + cy ** 2)
        gain = 1.0 + 0.18 * (1.0 - dist / max_d)
        gain += 0.05 * (1.0 - self._yy / max(1.0, self.H - 1))
        img *= gain[:, :, np.newaxis]

    def _draw_well(self, img: np.ndarray, cx: int, cy: int, state: int):
        """Draw one well: wall ring + coloured interior + highlight."""
        dist = np.sqrt((self._xx - cx) ** 2 + (self._yy - cy) ** 2)
        r    = self.well_radius

        # ── Well wall (reflective ring) ───────────────────────────────
        wall_mask = (dist >= r - 4) & (dist <= r + 2)
        img[wall_mask] = self.WALL_BGR

        # ── Interior ─────────────────────────────────────────────────
        inner_mask = dist < (r - 4)
        if not inner_mask.any():
            return

        if state < 0:
            self._draw_empty_well(img, inner_mask, dist, cx, cy, r)
            return

        base = self._positive_bgr.copy() if state > 0 else self._negative_bgr.copy()

        # Per-well colour variation (concentration / pipetting error)
        base += self.rng.normal(0, 6, 3).astype(np.float32)

        # Meniscus brightness: liquid is brightest near the wall (bowl shape)
        norm_dist = dist / (r - 4)
        brightness = np.where(inner_mask, 0.80 + 0.30 * np.clip(norm_dist, 0, 1), 1.0)

        for ch in range(3):
            img[:, :, ch][inner_mask] = base[ch] * brightness[inner_mask]

        # ── Specular reflection highlight (top-left quadrant) ─────────
        hx = cx - r * 0.38
        hy = cy - r * 0.38
        h_dist = np.sqrt((self._xx - hx) ** 2 + (self._yy - hy) ** 2)
        highlight_mask = inner_mask & (h_dist < r * 0.22)
        img[highlight_mask] = np.clip(img[highlight_mask] + 55.0, 0, 255)

    def _draw_empty_well(
        self,
        img: np.ndarray,
        inner_mask: np.ndarray,
        dist: np.ndarray,
        cx: int,
        cy: int,
        r: int,
    ):
        """Unused/control well: translucent grey with stronger reflections."""
        base = self._empty_bgr.copy()
        base += self.rng.normal(0, 4, 3).astype(np.float32)
        norm_dist = dist / max(1, r - 4)
        radial = np.where(inner_mask, 0.95 + 0.18 * np.clip(norm_dist, 0, 1), 1.0)

        for ch in range(3):
            current = img[:, :, ch][inner_mask]
            target = base[ch] * radial[inner_mask]
            img[:, :, ch][inner_mask] = 0.40 * current + 0.60 * target

        # Large reflective patches similar to clear wells in the photo.
        hx = cx + r * 0.10
        hy = cy - r * 0.12
        h_dist = np.sqrt((self._xx - hx) ** 2 + (self._yy - hy) ** 2)
        highlight_mask = inner_mask & (h_dist < r * 0.34)
        img[highlight_mask] = np.clip(img[highlight_mask] + 40.0, 0, 255)

        crescent = inner_mask & (self._xx > cx) & (self._yy < cy + r * 0.15)
        img[crescent] = np.clip(img[crescent] + 12.0, 0, 255)

    def _add_global_noise(self, img: np.ndarray):
        """Shot noise to simulate camera sensor."""
        noise = self.rng.normal(0, self.noise_std, img.shape).astype(np.float32)
        img += noise

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def show(self, image: np.ndarray, title: str = "Simulated Plate", ax=None):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 9))
        ax.imshow(rgb)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        # Annotate well positions
        row_labels = "ABCDEFGH"
        for ri, cy in enumerate(self.row_centers):
            for ci, cx in enumerate(self.col_centers):
                ax.text(cx, cy - self.well_radius - 3,
                        f"{row_labels[ri]}{ci+1}",
                        ha="center", va="bottom", fontsize=4.5,
                        color="white", alpha=0.7)
        return ax


def random_plate_labels(
    rows: int = 8,
    cols: int = 12,
    positive_fraction: float = 0.25,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a random 8×12 label matrix (1=purple, 0=blue), independent wells."""
    rng = np.random.RandomState(seed)
    return (rng.rand(rows, cols) < positive_fraction).astype(int)


def clustered_plate_labels(
    rows: int = 8,
    cols: int = 12,
    n_clusters: int = 3,
    cluster_sigma: float = 1.8,
    positive_fraction: float = 0.25,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a 8×12 label matrix where purple wells are spatially clustered.

    This better mimics real screening plates where active compounds share
    structural scaffolds and tend to cluster on the plate layout.

    Parameters
    ----------
    n_clusters      : number of hot-spot centres
    cluster_sigma   : spatial spread of each cluster (in well units)
    positive_fraction : target fraction of purple wells
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.RandomState(seed)

    # Place cluster centres randomly
    heat = np.zeros((rows, cols), dtype=float)
    for _ in range(n_clusters):
        cr = rng.randint(0, rows)
        cc = rng.randint(0, cols)
        heat[cr, cc] += 1.0

    # Spread each centre with a Gaussian kernel
    heat = gaussian_filter(heat, sigma=cluster_sigma, mode="constant")
    heat = heat / heat.max()      # normalise to [0, 1]

    # Convert to binary labels using a threshold that achieves ~positive_fraction
    threshold = np.percentile(heat, 100 * (1 - positive_fraction))
    labels = (heat >= threshold).astype(int)
    return labels
