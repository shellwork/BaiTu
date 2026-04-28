"""
OT-2 Battleship Closed-Loop Automated Experiment
=================================================
Orchestrates a full active-learning Battleship game on the OT-2:

1. **Setup** – generate a random board, OT-2 dispenses NaOH/H2O into 80 wells.
2. **Game loop** – model picks next well → OT-2 adds indicator → camera reads
   colour → update model → repeat until all ships sunk.
3. **Report** – save results, metrics, checkpoint.

Usage (run from repository root)
--------------------------------
  # Dry-run (synthetic images, no hardware)
  python -m hardware.battleship_ot2_loop --dry_run --strategy prob --seed 42

  # Real experiment
  python -m hardware.battleship_ot2_loop --strategy prob --seed 42 --output_dir run1
  python -m hardware.battleship_ot2_loop --strategy prob --seed 42 \\
      --robot_ip 169.254.200.128 --geometry_path hardware/calibration.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BOARD_COLS, BOARD_ROWS, DEFAULT_SHIP_SIZES, PLATE_COLS, PLATE_ROWS
from core.battleship_env import BattleshipBoard, Ship
from core.battleship_model import Game
from plate.battleship_plate_readout import (
    classify_ship_water_from_mean_bgr,
    load_image_bgr,
    query_well_fixed_geometry_rgb,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

# ── Default deck layout ──────────────────────────────────────────────
# The canonical deck configuration. Every field can be overridden at run
# time via ``LoopConfig.deck_overrides`` (from the dashboard) or via a
# JSON file passed through ``--deck_path``. Keep this dict as the single
# source of truth; don't re-introduce parallel module constants.
DEFAULT_DECK: Dict[str, object] = {
    # Pipette
    "pipette_name":  "p1000_single",
    "pipette_mount": "right",
    # Slot 1 — target 96-well plate
    "plate_slot":    "1",
    "plate_labware": "corning_96_wellplate_360ul_flat",
    # Slot 2 — tiprack (must match the pipette)
    "tiprack_slot":    "2",
    "tiprack_labware": "opentrons_96_tiprack_1000ul",
    # Slot 4 — NaOH reservoir (dispensed into empty cells)
    "naoh_slot":        "4",
    "naoh_labware":     "nest_12_reservoir_15ml",
    "naoh_source_well": "A1",
    # Slot 5 — H₂O reservoir (dispensed into ship cells)
    "h2o_slot":        "5",
    "h2o_labware":     "nest_12_reservoir_15ml",
    "h2o_source_well": "A1",
    # Slot 6 — indicator (cabbage juice) reservoir
    "indicator_slot":        "6",
    "indicator_labware":     "nest_12_reservoir_15ml",
    "indicator_source_well": "A1",
    # Volumes (µL)
    "fill_volume":      100.0,   # NaOH / H₂O per well, Phase 1
    "indicator_volume": 100.0,   # indicator per well,  Phase 2
}

# Z-offsets for liquid handling (mm) — not normally user-editable.
ASPIRATE_OFFSET = (0, 0, 1)    # aspirate: bottom + 1mm (closer to reservoir bottom)
DISPENSE_OFFSET = (0, 0, -1)   # dispense: top − 1mm   (just inside the well)

# Pipette tip capacity in µL — used to derive how many wells each aspirate covers.
TIP_CAPACITY_UL = 1000.0


@dataclass
class LoopConfig:
    # Strategy
    strategy: str = "prob"
    seed: Optional[int] = None

    # OT-2 connection
    robot_ip: str = "169.254.200.128"

    # Camera
    geometry_path: Optional[str] = None  # pre-calibrated geometry JSON
    color_develop_seconds: float = 10.0

    # Output
    output_dir: str = "ot2_loop_results"
    checkpoint_path: Optional[str] = None  # resume from checkpoint

    # Debug
    dry_run: bool = False
    skip_setup: bool = False   # skip Phase 1 (OT-2 init + board dispensing)

    # Error handling
    max_camera_retries: int = 3
    camera_retry_delay: float = 2.0

    # Human-in-the-loop colour QC
    # If |conf - 0.5| < human_check_margin the loop pauses and waits for an
    # operator to confirm hit/miss. Set to 0 to disable.
    human_check_margin: float = 0.05
    human_check_poll_seconds: float = 1.0
    human_check_timeout_seconds: float = 0.0   # 0 = wait forever

    # Deck layout: loaded from --deck_path JSON first, then merged with
    # deck_overrides. Final effective deck lives on OT2BattleshipLoop.deck.
    deck_path: Optional[str] = None
    deck_overrides: Dict[str, object] = field(default_factory=dict)


@dataclass
class StepRecord:
    step: int
    row: int
    col: int
    well_name: str
    label: str
    is_hit: bool
    sunk_ship_size: Optional[int]
    confidence: float
    mean_rgb: Tuple[int, int, int]
    image_path: str
    timestamp: str


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

ROW_LABELS = [chr(ord("A") + i) for i in range(PLATE_ROWS)]


def _rc_to_well(row: int, col: int) -> str:
    return f"{ROW_LABELS[row]}{col + 1}"


def _tip_well_name(index: int) -> str:
    """Map linear tip index (0-95) to well name on the tiprack."""
    r = index // 12
    c = index % 12
    return f"{chr(65 + r)}{c + 1}"


# ═══════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════

class OT2BattleshipLoop:

    def __init__(self, config: LoopConfig):
        self.cfg = config
        self.board: Optional[BattleshipBoard] = None
        self.model: Optional[Game] = None
        self.geometry: Optional[Dict] = None

        # Effective deck: defaults ← file ← overrides. Store on the instance
        # so every method reads from a single resolved dict.
        self.deck: Dict[str, object] = dict(DEFAULT_DECK)
        if config.deck_path:
            try:
                with open(config.deck_path) as f:
                    self.deck.update(json.load(f))
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("Could not read deck_path=%s: %s", config.deck_path, exc)
        if config.deck_overrides:
            self.deck.update(config.deck_overrides)

        fill_vol = float(self.deck["fill_volume"])
        self._max_wells_per_aspirate = max(1, int(TIP_CAPACITY_UL // fill_vol))

        self.results_matrix = np.full((PLATE_ROWS, PLATE_COLS), -1, dtype=int)
        self.history: List[StepRecord] = []
        self._tip_index = 0
        self._step = 0
        self._indicator_tip_held = False

        # Colour prototypes (loaded from calibration.json, or defaults)
        self._ship_rgb: Optional[np.ndarray] = None
        self._water_rgb: Optional[np.ndarray] = None
        self._rgb_l2_tolerance: float = 48.0

        # Optional LAB-space discriminant (set by calibration). When present
        # the classifier projects mean LAB onto a 1-D axis between the two
        # prototypes — much more robust than RGB L2 under fixed lighting.
        self._lab_ship: Optional[np.ndarray] = None
        self._lab_water: Optional[np.ndarray] = None
        self._lab_direction: Optional[np.ndarray] = None
        self._lab_threshold: Optional[float] = None
        self._lab_proto_distance: float = 0.0
        self._color_space: str = "rgb"   # "rgb" or "lab"

        # OT-2 labware IDs (populated by _setup_ot2)
        self._plate_id: Optional[str] = None
        self._tiprack_id: Optional[str] = None
        self._naoh_id: Optional[str] = None
        self._h2o_id: Optional[str] = None
        self._indicator_id: Optional[str] = None

        # Output directory
        self._out = Path(config.output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._images_dir = self._out / "images"
        self._images_dir.mkdir(exist_ok=True)

    # ── Phase 1: Initialisation ─────────────────────────────────────

    def _setup_ot2(self) -> None:
        """Create an OT-2 run and load all labware + pipette."""
        if self.cfg.dry_run:
            log.info("[DRY RUN] Skipping OT-2 setup")
            return

        from hardware.ot2_ctrl import OT2_functions as ot

        ot.ROBOT_IP = self.cfg.robot_ip
        _run_id, _ = ot.create_run()
        log.info("OT-2 run created: %s", _run_id)

        d = self.deck
        ot.load_equipment(0, d["pipette_name"])
        self._tiprack_id = ot.load_equipment(1, d["tiprack_labware"], d["tiprack_slot"])
        self._plate_id = ot.load_equipment(1, d["plate_labware"], d["plate_slot"])
        self._naoh_id = ot.load_equipment(1, d["naoh_labware"], d["naoh_slot"])
        self._h2o_id = ot.load_equipment(1, d["h2o_labware"], d["h2o_slot"])
        self._indicator_id = ot.load_equipment(1, d["indicator_labware"], d["indicator_slot"])

        log.info("Deck layout loaded:")
        log.info("  Slot %s: %s (tiprack)", d["tiprack_slot"], d["tiprack_labware"])
        log.info("  Slot %s: %s (plate)", d["plate_slot"], d["plate_labware"])
        log.info("  Slot %s: %s (NaOH)", d["naoh_slot"], d["naoh_labware"])
        log.info("  Slot %s: %s (H2O)", d["h2o_slot"], d["h2o_labware"])
        log.info("  Slot %s: %s (indicator)", d["indicator_slot"], d["indicator_labware"])
        log.info("  Pipette: %s (%s mount)", d["pipette_name"], d["pipette_mount"])

    def _setup_board(self) -> BattleshipBoard:
        """Generate board and dispense NaOH/H2O into all 80 active wells."""
        board = BattleshipBoard(
            rows=BOARD_ROWS,
            cols=BOARD_COLS,
            seed=self.cfg.seed,
        )
        log.info(
            "Board generated (seed=%s): %d ship cells",
            self.cfg.seed, board.total_ship_cells,
        )

        # grid=0 (empty) → NaOH well ; grid=1 (ship) → H₂O well
        naoh_wells: List[str] = []
        h2o_wells: List[str] = []
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                well_name = _rc_to_well(r, c)
                if board.grid[r, c] == 0:
                    naoh_wells.append(well_name)
                else:
                    h2o_wells.append(well_name)

        if self.cfg.dry_run:
            log.info(
                "[DRY RUN] Would dispense NaOH→%d wells, H2O→%d wells",
                len(naoh_wells), len(h2o_wells),
            )
        else:
            self._dispense_board_liquids(naoh_wells, h2o_wells)

        self.board = board
        return board

    def _transfer_liquid(
        self,
        source_id: str,
        source_well: str,
        dest_wells: List[str],
        volume: float,
        label: str,
    ) -> None:
        """Transfer liquid from one reservoir to multiple plate wells.

        Batch optimisation: aspirate N × volume at once from the reservoir,
        then dispense into N consecutive wells before returning for the next
        batch.  With 1000µL tips and 100µL/well this gives 10 wells per trip.
        """
        from hardware.ot2_ctrl import OT2_functions as ot

        tip_well = _tip_well_name(self._tip_index)
        self._tip_index += 1
        ot.pick_up(self._tiprack_id, tip_well)

        total = len(dest_wells)
        idx = 0
        while idx < total:
            # Determine batch size
            batch_size = min(self._max_wells_per_aspirate, total - idx)
            batch_wells = dest_wells[idx : idx + batch_size]
            aspirate_vol = volume * batch_size

            # Aspirate full batch from reservoir
            log.info("  Aspirate %.0f µL (%d×%.0f) from %s",
                     aspirate_vol, batch_size, volume, label)
            ot.move(source_id, source_well, offset=ASPIRATE_OFFSET)
            ot.aspirate(aspirate_vol, source_id, source_well,
                        offset=ASPIRATE_OFFSET, origin="bottom")

            # Dispense into each well in this batch
            for w in batch_wells:
                idx += 1
                log.info("  [%d/%d] dispense → plate:%s", idx, total, w)
                ot.move(self._plate_id, w, offset=DISPENSE_OFFSET)
                ot.dispense(volume, self._plate_id, w,
                            offset=DISPENSE_OFFSET, origin="top")

        ot.unload_to_trash()

    def _dispense_board_liquids(
        self,
        naoh_wells: List[str],
        h2o_wells: List[str],
    ) -> None:
        """OT-2 Phase 1: fill all 80 active wells with NaOH or H2O.

        Liquid mapping (matches _board_to_wells):
          - grid=0 (empty) → NaOH from slot 4 reservoir
          - grid=1 (ship)  → H₂O  from slot 5 reservoir
        """
        d = self.deck
        fill_vol = float(d["fill_volume"])
        # ── Step 1: NaOH from its slot → plate empty wells ──
        log.info("Dispensing NaOH (slot %s) → %d plate wells ...", d["naoh_slot"], len(naoh_wells))
        self._transfer_liquid(
            self._naoh_id, d["naoh_source_well"], naoh_wells,
            fill_vol, f"NaOH(slot{d['naoh_slot']}:{d['naoh_source_well']})",
        )

        # ── Step 2: H₂O from its slot → plate ship wells ──
        log.info("Dispensing H2O (slot %s) → %d plate wells ...", d["h2o_slot"], len(h2o_wells))
        self._transfer_liquid(
            self._h2o_id, d["h2o_source_well"], h2o_wells,
            fill_vol, f"H2O(slot{d['h2o_slot']}:{d['h2o_source_well']})",
        )

        log.info(
            "Phase 1 complete: NaOH(slot%s)→%d wells, H2O(slot%s)→%d wells",
            d["naoh_slot"], len(naoh_wells), d["h2o_slot"], len(h2o_wells),
        )

    def _calibrate_geometry(self) -> Dict:
        """Load geometry + optional colour prototypes from calibration file."""
        if self.cfg.geometry_path:
            with open(self.cfg.geometry_path) as f:
                calib = json.load(f)

            # Load colour prototypes if present in calibration file
            if "ship_rgb" in calib:
                self._ship_rgb = np.array(calib["ship_rgb"], dtype=np.float32)
                self._water_rgb = np.array(calib["water_rgb"], dtype=np.float32)
                self._rgb_l2_tolerance = calib.get("rgb_l2_tolerance", 48.0)
                log.info("Colour prototypes loaded:")
                log.info("  HIT  (ship)  RGB: %s", self._ship_rgb.tolist())
                log.info("  MISS (water) RGB: %s", self._water_rgb.tolist())
                log.info("  L2 tolerance: %.1f", self._rgb_l2_tolerance)

            # Optional LAB-space discriminant (preferred when present)
            if "lab_ship" in calib and "lab_water" in calib:
                self._lab_ship = np.array(calib["lab_ship"], dtype=np.float32)
                self._lab_water = np.array(calib["lab_water"], dtype=np.float32)
                diff = self._lab_ship - self._lab_water
                norm = float(np.linalg.norm(diff))
                if norm > 1e-6:
                    self._lab_direction = diff / norm
                    self._lab_threshold = float(
                        ((self._lab_ship + self._lab_water) / 2.0) @ self._lab_direction
                    )
                    self._lab_proto_distance = norm
                    self._color_space = "lab"
                    log.info("LAB discriminant loaded:")
                    log.info("  HIT  Lab: %s", self._lab_ship.tolist())
                    log.info("  MISS Lab: %s", self._lab_water.tolist())
                    log.info("  prototype distance: %.2f", norm)

            log.info("Loaded calibration from %s", self.cfg.geometry_path)
            return calib

        if self.cfg.dry_run:
            from plate.battleship_plate_simulation import get_fixed_well_geometry
            geo = get_fixed_well_geometry()
            log.info("[DRY RUN] Using synthetic well geometry")
            return geo

        # Real camera: capture one image and detect wells
        image_path = self._capture_image("calibration")
        from plate.plate_analysis import detect_wells_adaptive, build_geometry_from_detection

        result = detect_wells_adaptive(image_path)
        if result is None:
            raise RuntimeError(
                f"Geometry calibration failed on image: {image_path}. "
                "Try providing a pre-calibrated --geometry_path."
            )
        geo = build_geometry_from_detection(result)

        # Save for re-use
        geo_path = self._out / "geometry.json"
        with open(geo_path, "w") as f:
            json.dump(geo, f, indent=2)
        log.info("Geometry calibrated and saved to %s", geo_path)
        return geo

    # ── Phase 2: Game loop helpers ──────────────────────────────────

    def _get_next_position(self) -> Optional[Tuple[int, int]]:
        """Ask the model for the next well to query."""
        selectable_mask = np.ones((PLATE_ROWS, PLATE_COLS), dtype=bool)
        selectable_mask[:, BOARD_COLS:] = False  # columns 11-12 not selectable

        pos = self.model.select_query(
            self.cfg.strategy,
            grid_order=[
                (r, c) for r in range(PLATE_ROWS) for c in range(PLATE_COLS)
            ],
            allowed_cells={
                (r, c) for r in range(PLATE_ROWS) for c in range(BOARD_COLS)
            } - self.model.hits - self.model.misses - self.model.sunk_cells,
        )
        return pos

    def _pick_up_indicator_tip(self) -> None:
        """Pick up a tip for indicator dispensing (called once, kept across steps)."""
        if self.cfg.dry_run or self._indicator_tip_held:
            return
        from hardware.ot2_ctrl import OT2_functions as ot
        tip_well = _tip_well_name(self._tip_index)
        self._tip_index += 1
        ot.pick_up(self._tiprack_id, tip_well)
        self._indicator_tip_held = True
        log.info("Picked up indicator tip: %s", tip_well)

    def _dispense_indicator(self, well_name: str) -> None:
        """Phase 2: dispense indicator into one well (reuse same tip)."""
        if self.cfg.dry_run:
            log.info("[DRY RUN] Dispense indicator → %s", well_name)
            return

        from hardware.ot2_ctrl import OT2_functions as ot

        self._pick_up_indicator_tip()

        d = self.deck
        indicator_src = d["indicator_source_well"]
        indicator_vol = float(d["indicator_volume"])
        log.info("  Indicator: slot%s:%s → plate:%s",
                 d["indicator_slot"], indicator_src, well_name)
        # Move to reservoir, then aspirate
        ot.move(self._indicator_id, indicator_src, offset=ASPIRATE_OFFSET)
        ot.aspirate(indicator_vol, self._indicator_id, indicator_src,
                    offset=ASPIRATE_OFFSET, origin="bottom")
        # Move to plate well, then dispense
        ot.move(self._plate_id, well_name, offset=DISPENSE_OFFSET)
        ot.dispense(indicator_vol, self._plate_id, well_name,
                    offset=DISPENSE_OFFSET, origin="top")

    def _park_arm(self) -> None:
        """Move arm to indicator reservoir (slot 6) to clear the camera view."""
        if self.cfg.dry_run:
            return
        from hardware.ot2_ctrl import OT2_functions as ot
        if self._indicator_id:
            ot.move(self._indicator_id, self.deck["indicator_source_well"])

    def _reset_ot2(self) -> None:
        """Safety reset: home first (raise to top), then drop tip."""
        if self.cfg.dry_run:
            log.info("[DRY RUN] Reset OT-2 (home + drop tip)")
            return

        from hardware.ot2_ctrl import OT2_functions as ot

        # 1) Home first — raise pipette to top to avoid collisions
        ot.home()
        log.info("OT-2 homed (pipette raised to top)")

        # 2) Then drop tip (ignore error if no tip is attached)
        try:
            ot.unload_to_trash()
            log.info("Tip dropped to trash")
        except Exception:
            log.info("No tip to drop (or already dropped)")

        # 3) Home again after dropping tip
        ot.home()
        log.info("OT-2 reset complete")

    def _capture_image(self, tag: str) -> str:
        """Capture a plate image. Returns the saved file path.

        Uses cv2.VideoCapture directly to avoid path issues in Helper.
        Images are saved to self._images_dir/<tag>_<timestamp>.jpg.
        """
        if self.cfg.dry_run:
            return self._capture_synthetic_image(tag)

        for attempt in range(self.cfg.max_camera_retries):
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    raise RuntimeError("Cannot open camera")

                # Drop a few frames for exposure stabilisation
                for _ in range(10):
                    cap.read()

                ret, frame = cap.read()
                cap.release()

                if not ret or frame is None:
                    raise RuntimeError("Failed to read frame from camera")

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = str(self._images_dir / f"{tag}_{ts}.jpg")
                cv2.imwrite(save_path, frame)
                log.info("Image captured: %s", save_path)
                return save_path

            except Exception as e:
                log.warning("Camera attempt %d failed: %s", attempt + 1, e)
                time.sleep(self.cfg.camera_retry_delay)

        raise RuntimeError(
            f"Camera failed after {self.cfg.max_camera_retries} retries"
        )

    def _capture_synthetic_image(self, tag: str) -> str:
        """Generate a synthetic plate image for dry-run mode."""
        from plate.battleship_plate_simulation import simulate_photo_from_board

        img = simulate_photo_from_board(self.board, seed=self.cfg.seed)
        path = str(self._images_dir / f"{tag}.png")
        cv2.imwrite(path, img)
        return path

    def _ensure_geometry(self, image_path: str) -> None:
        """Lazy geometry calibration: detect wells from the first real photo."""
        if self.geometry is not None:
            return

        log.info("First photo with liquid — calibrating geometry ...")
        from plate.plate_analysis import detect_wells_adaptive, build_geometry_from_detection

        result = detect_wells_adaptive(image_path)
        if result is None:
            raise RuntimeError(
                f"Geometry calibration failed on image: {image_path}. "
                "Try providing a pre-calibrated --geometry_path."
            )
        self.geometry = build_geometry_from_detection(result)

        geo_path = self._out / "geometry.json"
        with open(geo_path, "w") as f:
            json.dump(self.geometry, f, indent=2)
        log.info("Geometry calibrated and saved to %s", geo_path)

    def _classify_well(
        self,
        image_path: str,
        row: int,
        col: int,
    ) -> Tuple[str, float, Tuple[int, int, int]]:
        """Read one well colour from the plate image.

        Uses calibrated LAB-space discriminant when available, otherwise the
        legacy RGB nearest-prototype classifier. Always returns a label
        (never "unknown") — a low-confidence call is still better than none,
        and ambiguous calls are caught later by the human-check trigger.
        """
        if self.cfg.dry_run:
            is_hit = bool(self.board.grid[row, col])
            label = "ship" if is_hit else "water"
            return label, 1.0, (0, 0, 0)

        # Calibrate geometry on first real photo (liquid already present)
        self._ensure_geometry(image_path)

        from plate.battleship_plate_readout import (
            mean_bgr_to_mean_rgb,
            sample_well_mean_bgr,
        )

        image_bgr = load_image_bgr(image_path)
        mean_bgr = sample_well_mean_bgr(image_bgr, row, col, self.geometry)
        mean_rgb = mean_bgr_to_mean_rgb(mean_bgr)
        rgb = (int(mean_rgb[0]), int(mean_rgb[1]), int(mean_rgb[2]))

        # Preferred path: 1-D LAB discriminant from calibration
        if self._color_space == "lab" and self._lab_direction is not None:
            mean_lab = self._bgr_to_lab(mean_bgr)
            projection = float(mean_lab @ self._lab_direction) - self._lab_threshold
            half_span = max(self._lab_proto_distance / 2.0, 1e-6)
            margin = float(np.tanh(abs(projection) / half_span))   # 0 at threshold, →1 at prototype
            conf = 0.5 + 0.5 * margin
            label = "ship" if projection >= 0 else "water"
            log.info(
                "    %s RGB=(%d,%d,%d) Lab=(%.1f,%.1f,%.1f) proj=%+.2f → %s (%.2f)",
                _rc_to_well(row, col), *rgb,
                float(mean_lab[0]), float(mean_lab[1]), float(mean_lab[2]),
                projection, label, conf,
            )
            return label, conf, rgb

        # Legacy RGB nearest-prototype path
        if self._ship_rgb is not None:
            ship_rgb = self._ship_rgb
            water_rgb = self._water_rgb
        else:
            if not hasattr(self, "_color_warned"):
                log.warning(
                    "No calibrated colour prototypes! Using defaults. "
                    "Run: python -m hardware.calibrate_geometry annotate <photo> "
                    "and use --geometry_path calibration.json"
                )
                self._color_warned = True
            from plate.battleship_plate_readout import SHIP_LIQUID_RGB, WATER_LIQUID_RGB
            ship_rgb = SHIP_LIQUID_RGB
            water_rgb = WATER_LIQUID_RGB

        dist_ship = float(np.linalg.norm(mean_rgb - ship_rgb))
        dist_water = float(np.linalg.norm(mean_rgb - water_rgb))

        if dist_ship <= dist_water:
            label = "ship"
            conf = max(0.0, min(1.0, dist_water / (dist_ship + dist_water + 1e-9)))
        else:
            label = "water"
            conf = max(0.0, min(1.0, dist_ship / (dist_ship + dist_water + 1e-9)))

        log.info(
            "    %s RGB=(%d,%d,%d) dist_hit=%.1f dist_miss=%.1f → %s (%.2f)",
            _rc_to_well(row, col), *rgb, dist_ship, dist_water, label, conf,
        )

        return label, conf, rgb

    @staticmethod
    def _bgr_to_lab(mean_bgr: np.ndarray) -> np.ndarray:
        """Convert a single mean BGR triple to OpenCV's 8-bit Lab space."""
        bgr_u8 = np.clip(mean_bgr.reshape(1, 1, 3), 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)
        return lab.reshape(3).astype(np.float32)

    # ── Human-in-the-loop QC (file-based IPC) ───────────────────────

    def _human_check_request_path(self) -> Path:
        return self._out / "human_check_request.json"

    def _human_check_response_path(self) -> Path:
        return self._out / "human_check_response.json"

    def _corrections_path(self) -> Path:
        return self._out / "corrections.json"

    def _maybe_human_check(
        self,
        row: int,
        col: int,
        label: str,
        conf: float,
        rgb: Tuple[int, int, int],
        image_path: str,
    ) -> Tuple[str, float, bool]:
        """If conf is too close to 0.5, pause for an operator confirmation.

        Returns (label, conf, was_overridden). Called between classification
        and model update.
        """
        margin = self.cfg.human_check_margin
        if margin <= 0:
            return label, conf, False
        if abs(conf - 0.5) >= margin:
            return label, conf, False

        req = {
            "step": self._step,
            "row": int(row),
            "col": int(col),
            "well": _rc_to_well(row, col),
            "auto_label": label,
            "confidence": float(conf),
            "mean_rgb": list(rgb),
            "image_path": str(image_path),
            "requested_at": datetime.now().isoformat(),
        }
        req_path = self._human_check_request_path()
        resp_path = self._human_check_response_path()
        # Clear any stale response from a previous request
        try:
            resp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except TypeError:  # py<3.8 safety net (unused here)
            if resp_path.exists():
                resp_path.unlink()

        with open(req_path, "w") as f:
            json.dump(req, f, indent=2)

        log.warning(
            "  ⚠ Ambiguous reading (conf=%.3f) for %s — waiting for human check ...",
            conf, _rc_to_well(row, col),
        )

        deadline = None
        if self.cfg.human_check_timeout_seconds > 0:
            deadline = time.time() + self.cfg.human_check_timeout_seconds

        while True:
            if resp_path.exists():
                try:
                    with open(resp_path) as f:
                        resp = json.load(f)
                    new_label = str(resp.get("label", label))
                    if new_label not in ("ship", "water"):
                        new_label = label
                    overridden = new_label != label
                    log.info(
                        "  ✔ Human decision: %s (was auto=%s, conf=%.3f)",
                        new_label, label, conf,
                    )
                    # Cleanup
                    resp_path.unlink()
                    if req_path.exists():
                        req_path.unlink()
                    return new_label, 1.0 if overridden else conf, overridden
                except (json.JSONDecodeError, OSError):
                    # Mid-write: retry
                    pass
            if deadline is not None and time.time() > deadline:
                log.warning(
                    "  ⚠ Human check timed out — keeping auto label '%s'", label,
                )
                if req_path.exists():
                    req_path.unlink()
                return label, conf, False
            time.sleep(self.cfg.human_check_poll_seconds)

    def _apply_pending_corrections(self) -> None:
        """Read corrections.json (if present) and rebuild model state.

        Each correction has the form ``{"row": int, "col": int, "label": "ship"|"water"}``.
        After a correction is applied the file is consumed (deleted) so the
        same edit isn't re-applied on every step.
        """
        path = self._corrections_path()
        if not path.exists():
            return
        try:
            with open(path) as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read %s: %s", path, exc)
            return

        items = payload.get("corrections", []) if isinstance(payload, dict) else payload
        if not items:
            try:
                path.unlink()
            except OSError:
                pass
            return

        applied = 0
        for item in items:
            try:
                r = int(item["row"])
                c = int(item["col"])
                new_label = str(item["label"])
                if new_label not in ("ship", "water"):
                    continue
            except (KeyError, TypeError, ValueError):
                continue

            target_idx = None
            for i, rec in enumerate(self.history):
                if rec.row == r and rec.col == c:
                    target_idx = i
                    break
            if target_idx is None:
                log.warning(
                    "Correction for %s ignored — well not in history.",
                    _rc_to_well(r, c),
                )
                continue

            old = self.history[target_idx]
            new_is_hit = (new_label == "ship")
            self.history[target_idx] = StepRecord(
                step=old.step, row=old.row, col=old.col, well_name=old.well_name,
                label=new_label, is_hit=new_is_hit,
                sunk_ship_size=old.sunk_ship_size,
                confidence=1.0,
                mean_rgb=old.mean_rgb,
                image_path=old.image_path,
                timestamp=datetime.now().isoformat(),
            )
            self.results_matrix[r, c] = 1 if new_is_hit else 0
            applied += 1
            log.info(
                "  ✏ Manual correction applied: %s → %s",
                _rc_to_well(r, c), new_label,
            )

        if applied:
            self._rebuild_model_from_history()
            self._save_checkpoint()

        try:
            path.unlink()
        except OSError:
            pass

    def _rebuild_model_from_history(self) -> None:
        """Recreate board+model from seed and replay history.

        Used after manual corrections so the belief reflects the corrected
        labels for every subsequent acquisition.
        """
        if self.cfg.seed is None and self.board is None:
            return
        seed = self.cfg.seed
        self.board = BattleshipBoard(rows=BOARD_ROWS, cols=BOARD_COLS, seed=seed)
        self.model = Game(board_rows=BOARD_ROWS, board_cols=BOARD_COLS)
        for rec in self.history:
            # Replay against the ground-truth board so sunk-ship info is
            # consistent with the observed labels.
            _, sunk = self.board.query(rec.row, rec.col)
            self.model.update(rec.row, rec.col, is_hit=rec.is_hit, sunk_ship=sunk)

    # ── Checkpoint persistence ──────────────────────────────────────

    def _save_checkpoint(self) -> None:
        cp = {
            "step": self._step,
            "tip_index": self._tip_index,
            "seed": self.cfg.seed,
            "strategy": self.cfg.strategy,
            "results_matrix": self.results_matrix.tolist(),
            "history": [asdict(r) for r in self.history],
        }
        path = self._out / "checkpoint.json"
        with open(path, "w") as f:
            json.dump(cp, f, indent=2)

    def _load_checkpoint(self) -> bool:
        cp_path = self.cfg.checkpoint_path
        if cp_path is None:
            return False
        if not os.path.exists(cp_path):
            log.warning("Checkpoint not found: %s", cp_path)
            return False

        with open(cp_path) as f:
            cp = json.load(f)

        self._step = cp["step"]
        self._tip_index = cp["tip_index"]
        self.results_matrix = np.array(cp["results_matrix"], dtype=int)
        self.history = [
            StepRecord(**{k: tuple(v) if k == "mean_rgb" else v for k, v in r.items()})
            for r in cp["history"]
        ]

        # Rebuild board from same seed
        self.board = BattleshipBoard(
            rows=BOARD_ROWS, cols=BOARD_COLS, seed=cp["seed"],
        )
        # Replay queries into board
        for rec in self.history:
            self.board.query(rec.row, rec.col)

        # Rebuild model
        self.model = Game(board_rows=BOARD_ROWS, board_cols=BOARD_COLS)
        for rec in self.history:
            sunk = None
            if rec.sunk_ship_size is not None:
                for s in self.board.ships:
                    if s.size == rec.sunk_ship_size and s.is_sunk():
                        sunk = s
                        break
            self.model.update(rec.row, rec.col, is_hit=rec.is_hit, sunk_ship=sunk)

        log.info("Resumed from checkpoint: step=%d", self._step)
        return True

    # ── Final results ───────────────────────────────────────────────

    def _save_final_results(self) -> Dict:
        total_queries = len(self.history)
        total_ships = self.board.total_ship_cells
        hits = sum(1 for r in self.history if r.is_hit)
        unknowns = sum(1 for r in self.history if r.label == "unknown")

        metrics = {
            "total_queries": total_queries,
            "total_ship_cells": total_ships,
            "hits_found": hits,
            "ships_sunk": len(self.board.get_sunk_ships()),
            "ships_total": len(self.board.ships),
            "unknown_classifications": unknowns,
            "strategy": self.cfg.strategy,
            "seed": self.cfg.seed,
            "game_over": self.board.is_game_over(),
        }

        # Save results
        results_path = self._out / "results.json"
        with open(results_path, "w") as f:
            json.dump({
                "config": asdict(self.cfg),
                "metrics": metrics,
                "history": [asdict(r) for r in self.history],
                "ground_truth": self.board.grid.tolist(),
                "final_results_matrix": self.results_matrix.tolist(),
            }, f, indent=2)

        np.save(str(self._out / "results_matrix.npy"), self.results_matrix)
        np.save(str(self._out / "ground_truth.npy"), self.board.grid)

        log.info("="*55)
        log.info("EXPERIMENT COMPLETE")
        log.info("="*55)
        log.info("  Strategy:    %s", self.cfg.strategy)
        log.info("  Seed:        %s", self.cfg.seed)
        log.info("  Queries:     %d", total_queries)
        log.info("  Hits found:  %d / %d", hits, total_ships)
        log.info("  Ships sunk:  %d / %d", len(self.board.get_sunk_ships()), len(self.board.ships))
        log.info("  Game over:   %s", self.board.is_game_over())
        log.info("  Output:      %s", self._out)
        log.info("="*55)

        return metrics

    # ── Main entry ──────────────────────────────────────────────────

    def run(self) -> Dict:
        """Execute the full closed-loop experiment."""
        resumed = self._load_checkpoint()

        if not resumed:
            if self.cfg.skip_setup:
                log.info("--skip_setup: skipping OT-2 init and board dispensing")
                # Still need OT-2 connection + labware IDs for game loop
                if not self.cfg.dry_run:
                    from hardware.ot2_ctrl import OT2_functions as ot
                    ot.ROBOT_IP = self.cfg.robot_ip
                    ot.reconnect_last_run()
                    # Recover labware IDs from the existing run
                    d = self.deck
                    self._tiprack_id = ot.get_labware_id_by_slot(d["tiprack_slot"])
                    self._plate_id = ot.get_labware_id_by_slot(d["plate_slot"])
                    self._indicator_id = ot.get_labware_id_by_slot(d["indicator_slot"])
                # Generate board (for ground truth) without dispensing
                self.board = BattleshipBoard(
                    rows=BOARD_ROWS, cols=BOARD_COLS, seed=self.cfg.seed,
                )
                log.info("Board generated (seed=%s): %d ship cells",
                         self.cfg.seed, self.board.total_ship_cells)
            else:
                self._setup_ot2()
                self._setup_board()
            self.model = Game(board_rows=BOARD_ROWS, board_cols=BOARD_COLS)

        # Geometry is calibrated lazily — on the first photo after
        # coloured liquid has been dispensed (not on the empty plate).
        if self.cfg.geometry_path:
            self.geometry = self._calibrate_geometry()

        log.info("Starting game loop (strategy=%s) ...", self.cfg.strategy)

        try:
            while not self.board.is_game_over():
                # 0) Apply any pending manual corrections from the dashboard
                self._apply_pending_corrections()

                # 1) Model decision
                pos = self._get_next_position()
                if pos is None:
                    log.info("Model returned no position — stopping.")
                    break

                row, col = pos
                well_name = _rc_to_well(row, col)
                self._step += 1
                log.info(
                    "[Step %d] Query %s (row=%d, col=%d)",
                    self._step, well_name, row, col,
                )

                # 2) Dispense indicator
                self._dispense_indicator(well_name)

                # 3) Move arm to slot 6 (indicator reservoir) to clear camera view
                self._park_arm()

                # 4) Wait for colour development
                if not self.cfg.dry_run:
                    log.info("  Waiting %.0fs for colour development ...", self.cfg.color_develop_seconds)
                    time.sleep(self.cfg.color_develop_seconds)

                # 5) Capture image and classify
                image_path = self._capture_image(f"step_{self._step:03d}")
                label, conf, mean_rgb = self._classify_well(image_path, row, col)

                # 5b) Human check for ambiguous readings
                label, conf, overridden = self._maybe_human_check(
                    row, col, label, conf, mean_rgb, image_path,
                )
                is_hit = (label == "ship")

                # 6) Update ground truth (get sunk info)
                _actual_hit, sunk_ship = self.board.query(row, col)

                # 7) Update model
                self.model.update(row, col, is_hit=is_hit, sunk_ship=sunk_ship)

                # 8) Update results matrix
                self.results_matrix[row, col] = 1 if is_hit else 0

                # 9) Record
                self.history.append(StepRecord(
                    step=self._step,
                    row=row,
                    col=col,
                    well_name=well_name,
                    label=label,
                    is_hit=is_hit,
                    sunk_ship_size=sunk_ship.size if sunk_ship else None,
                    confidence=conf,
                    mean_rgb=mean_rgb,
                    image_path=image_path,
                    timestamp=datetime.now().isoformat(),
                ))

                sunk_str = f"  ★ SUNK ship (size {sunk_ship.size})!" if sunk_ship else ""
                log.info(
                    "  → %s (conf=%.2f) %s  [%d/%d ships sunk]%s",
                    "HIT" if is_hit else "MISS",
                    conf,
                    well_name,
                    len(self.board.get_sunk_ships()),
                    len(self.board.ships),
                    sunk_str,
                )

                # 9) Checkpoint
                self._save_checkpoint()

        except Exception as e:
            log.error("Experiment interrupted: %s", e)
            self._save_checkpoint()
            self._reset_ot2()
            raise

        # Normal finish: reset robot and save results
        self._reset_ot2()
        return self._save_final_results()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="OT-2 Battleship Closed-Loop Experiment",
    )
    parser.add_argument("--strategy", default="prob",
                        choices=["prob", "entropy", "hunt_target", "pro_solver", "random"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--robot_ip", default="169.254.200.128")
    parser.add_argument("--color_develop_seconds", type=float, default=10.0)
    parser.add_argument("--geometry_path", default=None)
    parser.add_argument("--output_dir", default="ot2_loop_results")
    parser.add_argument("--resume", default=None, help="checkpoint JSON path")
    parser.add_argument("--dry_run", action="store_true",
                        help="simulate without OT-2 hardware")
    parser.add_argument("--skip_setup", action="store_true",
                        help="skip Phase 1 (OT-2 init + board dispensing), jump to game loop")
    parser.add_argument("--deck_path", default=None,
                        help="JSON file overriding DEFAULT_DECK (slots / labware / "
                             "source wells / volumes). Keys missing from the file "
                             "fall back to DEFAULT_DECK.")
    parser.add_argument("--human_check_margin", type=float, default=0.05,
                        help="Pause for operator confirmation when "
                             "|conf - 0.5| < margin. Default 0.05; set to 0 to "
                             "disable.")
    parser.add_argument("--human_check_timeout_seconds", type=float, default=0.0,
                        help="Give up on a human-check prompt and keep the auto "
                             "label after this many seconds. 0 = wait forever.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    config = LoopConfig(
        strategy=args.strategy,
        seed=args.seed,
        robot_ip=args.robot_ip,
        color_develop_seconds=args.color_develop_seconds,
        geometry_path=args.geometry_path,
        output_dir=args.output_dir,
        checkpoint_path=args.resume,
        dry_run=args.dry_run,
        skip_setup=args.skip_setup,
        deck_path=args.deck_path,
        human_check_margin=args.human_check_margin,
        human_check_timeout_seconds=args.human_check_timeout_seconds,
    )

    loop = OT2BattleshipLoop(config)
    loop.run()


def reset():
    """Standalone reset: home (raise to top) → drop tip → home again."""
    parser = argparse.ArgumentParser(description="Reset OT-2 (home + drop tip)")
    parser.add_argument("--robot_ip", default="169.254.200.128")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from hardware.ot2_ctrl import OT2_functions as ot
    ot.ROBOT_IP = args.robot_ip
    ot.reconnect_last_run()

    # 1) Home first — raise pipette to avoid collisions
    ot.home()
    log.info("OT-2 homed (pipette raised to top)")

    # 2) Drop tip if attached
    try:
        ot.unload_to_trash()
        log.info("Tip dropped to trash")
    except Exception:
        log.info("No tip to drop")

    # 3) Home again
    ot.home()
    log.info("OT-2 reset complete")


def calibrate():
    """Capture an image and attempt geometry detection.

    Saves the image and (if successful) geometry.json for inspection.
    Usage:
        python battleship_ot2_loop.py calibrate
        python battleship_ot2_loop.py calibrate --output_dir my_calib
    """
    parser = argparse.ArgumentParser(description="Calibrate plate geometry from camera")
    parser.add_argument("--output_dir", default="ot2_loop_results")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Capture image
    log.info("Capturing plate image ...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open camera")
        return
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        log.error("Failed to read frame")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = str(out / f"calibration_{ts}.jpg")
    cv2.imwrite(img_path, frame)
    log.info("Image saved: %s", img_path)
    log.info("Image shape: %s", frame.shape)

    # Try detection
    from plate.plate_analysis import detect_wells_adaptive, build_geometry_from_detection
    result = detect_wells_adaptive(img_path)

    if result is None:
        log.warning("Auto-detection FAILED.")
        log.warning("Check the image: open %s", img_path)
        log.warning("Possible fixes:")
        log.warning("  1. Adjust camera position / distance / focus")
        log.warning("  2. Improve lighting (avoid reflections)")
        log.warning("  3. Fill some wells with coloured liquid for better contrast")
        log.warning("  4. Manually create geometry.json (see docs)")
        return

    geo = build_geometry_from_detection(result)
    geo_path = str(out / "geometry.json")
    with open(geo_path, "w") as f:
        json.dump(geo, f, indent=2)

    log.info("Geometry calibration SUCCESS!")
    log.info("  Saved to: %s", geo_path)
    log.info("  Row centers: %s", geo["row_centers"])
    log.info("  Col centers: %s", geo["col_centers"])
    log.info("  Well radius: %d px", geo["well_radius"])
    log.info("")
    log.info("Use it with:  python battleship_ot2_loop.py --geometry_path %s ...", geo_path)


if __name__ == "__main__":
    # python battleship_ot2_loop.py reset      → reset only
    # python battleship_ot2_loop.py calibrate   → camera calibration
    # python battleship_ot2_loop.py [...]       → run experiment
    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        sys.argv.pop(1)
        reset()
    elif len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        sys.argv.pop(1)
        calibrate()
    else:
        main()
