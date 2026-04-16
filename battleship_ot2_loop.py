"""
OT-2 Battleship Closed-Loop Automated Experiment
=================================================
Orchestrates a full active-learning Battleship game on the OT-2:

1. **Setup** – generate a random board, OT-2 dispenses NaOH/H2O into 80 wells.
2. **Game loop** – model picks next well → OT-2 adds indicator → camera reads
   colour → update model → repeat until all ships sunk.
3. **Report** – save results, metrics, checkpoint.

Usage
-----
  # Dry-run (synthetic images, no hardware)
  python battleship_ot2_loop.py --dry_run --strategy prob --seed 42

  # Real experiment
  python battleship_ot2_loop.py --strategy prob --seed 42 --output_dir run1
  python battleship_ot2_loop.py --strategy prob --seed 42 --robot_ip 169.254.200.128
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

from config import BOARD_COLS, BOARD_ROWS, DEFAULT_SHIP_SIZES, PLATE_COLS, PLATE_ROWS
from battleship_env import BattleshipBoard, Ship
from battleship_model import Game
from plate.battleship_plate_readout import (
    classify_ship_water_from_mean_bgr,
    load_image_bgr,
    query_well_fixed_geometry_rgb,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

# ── Fixed deck layout ────────────────────────────────────────────────
# Pipette
PIPETTE_NAME = "p1000_single"
PIPETTE_MOUNT = "right"
# Slot 1: 96-well plate (target)
PLATE_SLOT = "1"
PLATE_LABWARE = "corning_96_wellplate_360ul_flat"
# Slot 2: tiprack (must match pipette — p1000 requires 1000µL tips)
TIPRACK_SLOT = "2"
TIPRACK_LABWARE = "opentrons_96_tiprack_1000ul"
# Slot 4: NaOH reservoir
NAOH_SLOT = "4"
NAOH_LABWARE = "nest_12_reservoir_15ml"
NAOH_SOURCE_WELL = "A1"
# Slot 5: H₂O reservoir
H2O_SLOT = "5"
H2O_LABWARE = "nest_12_reservoir_15ml"
H2O_SOURCE_WELL = "A1"
# Slot 6: indicator (cabbage juice) reservoir
INDICATOR_SLOT = "6"
INDICATOR_LABWARE = "nest_12_reservoir_15ml"
INDICATOR_SOURCE_WELL = "A1"

# Fixed volumes (µL)
FILL_VOLUME = 100.0       # NaOH / H₂O per well (Phase 1: board setup)
INDICATOR_VOLUME = 100.0   # indicator per well   (Phase 2: game loop)

# Z-offsets for liquid handling (mm)
ASPIRATE_OFFSET = (0, 0, 1)    # aspirate: bottom + 1mm (closer to reservoir bottom)
DISPENSE_OFFSET = (0, 0, -1)   # dispense: top − 1mm   (just inside the well)

# Batch aspirate: 1000µL tip ÷ 100µL/well = 10 wells per aspirate
MAX_WELLS_PER_ASPIRATE = int(1000 // FILL_VOLUME)  # = 10


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

        self.results_matrix = np.full((PLATE_ROWS, PLATE_COLS), -1, dtype=int)
        self.history: List[StepRecord] = []
        self._tip_index = 0
        self._step = 0
        self._indicator_tip_held = False

        # Colour prototypes (loaded from calibration.json, or defaults)
        self._ship_rgb: Optional[np.ndarray] = None
        self._water_rgb: Optional[np.ndarray] = None
        self._rgb_l2_tolerance: float = 48.0

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

        from BaituOT2Battleship.OT2_Ctrl import OT2_functions as ot

        ot.ROBOT_IP = self.cfg.robot_ip
        _run_id, _ = ot.create_run()
        log.info("OT-2 run created: %s", _run_id)

        ot.load_equipment(0, PIPETTE_NAME)
        self._tiprack_id = ot.load_equipment(1, TIPRACK_LABWARE, TIPRACK_SLOT)
        self._plate_id = ot.load_equipment(1, PLATE_LABWARE, PLATE_SLOT)
        self._naoh_id = ot.load_equipment(1, NAOH_LABWARE, NAOH_SLOT)
        self._h2o_id = ot.load_equipment(1, H2O_LABWARE, H2O_SLOT)
        self._indicator_id = ot.load_equipment(1, INDICATOR_LABWARE, INDICATOR_SLOT)

        log.info("Deck layout loaded:")
        log.info("  Slot %s: %s (tiprack)", TIPRACK_SLOT, TIPRACK_LABWARE)
        log.info("  Slot %s: %s (plate)", PLATE_SLOT, PLATE_LABWARE)
        log.info("  Slot %s: %s (NaOH)", NAOH_SLOT, NAOH_LABWARE)
        log.info("  Slot %s: %s (H2O)", H2O_SLOT, H2O_LABWARE)
        log.info("  Slot %s: %s (indicator)", INDICATOR_SLOT, INDICATOR_LABWARE)
        log.info("  Pipette: %s (%s mount)", PIPETTE_NAME, PIPETTE_MOUNT)

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

        from plate.battleship_plate_no_touch import _board_to_wells

        # _board_to_wells expects list-of-lists with 'x'/'o'
        char_board = [
            ["o" if board.grid[r, c] == 1 else "x" for c in range(BOARD_COLS)]
            for r in range(BOARD_ROWS)
        ]
        naoh_wells, h2o_wells = _board_to_wells(char_board)

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
        from BaituOT2Battleship.OT2_Ctrl import OT2_functions as ot

        tip_well = _tip_well_name(self._tip_index)
        self._tip_index += 1
        ot.pick_up(self._tiprack_id, tip_well)

        total = len(dest_wells)
        idx = 0
        while idx < total:
            # Determine batch size
            batch_size = min(MAX_WELLS_PER_ASPIRATE, total - idx)
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
        # ── Step 1: NaOH from Slot 4 → plate empty wells ──
        log.info("Dispensing NaOH (slot %s) → %d plate wells ...", NAOH_SLOT, len(naoh_wells))
        self._transfer_liquid(
            self._naoh_id, NAOH_SOURCE_WELL, naoh_wells,
            FILL_VOLUME, f"NaOH(slot{NAOH_SLOT}:{NAOH_SOURCE_WELL})",
        )

        # ── Step 2: H₂O from Slot 5 → plate ship wells ──
        log.info("Dispensing H2O (slot %s) → %d plate wells ...", H2O_SLOT, len(h2o_wells))
        self._transfer_liquid(
            self._h2o_id, H2O_SOURCE_WELL, h2o_wells,
            FILL_VOLUME, f"H2O(slot{H2O_SLOT}:{H2O_SOURCE_WELL})",
        )

        log.info("Phase 1 complete: NaOH(slot%s)→%d wells, H2O(slot%s)→%d wells",
                 NAOH_SLOT, len(naoh_wells), H2O_SLOT, len(h2o_wells))

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
        from BaituOT2Battleship.OT2_Ctrl import OT2_functions as ot
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

        from BaituOT2Battleship.OT2_Ctrl import OT2_functions as ot

        self._pick_up_indicator_tip()

        log.info("  Indicator: slot%s:%s → plate:%s",
                 INDICATOR_SLOT, INDICATOR_SOURCE_WELL, well_name)
        # Move to reservoir, then aspirate
        ot.move(self._indicator_id, INDICATOR_SOURCE_WELL, offset=ASPIRATE_OFFSET)
        ot.aspirate(INDICATOR_VOLUME, self._indicator_id, INDICATOR_SOURCE_WELL,
                    offset=ASPIRATE_OFFSET, origin="bottom")
        # Move to plate well, then dispense
        ot.move(self._plate_id, well_name, offset=DISPENSE_OFFSET)
        ot.dispense(INDICATOR_VOLUME, self._plate_id, well_name,
                    offset=DISPENSE_OFFSET, origin="top")

    def _park_arm(self) -> None:
        """Move arm to indicator reservoir (slot 6) to clear the camera view."""
        if self.cfg.dry_run:
            return
        from BaituOT2Battleship.OT2_Ctrl import OT2_functions as ot
        if self._indicator_id:
            ot.move(self._indicator_id, INDICATOR_SOURCE_WELL)

    def _reset_ot2(self) -> None:
        """Safety reset: home first (raise to top), then drop tip."""
        if self.cfg.dry_run:
            log.info("[DRY RUN] Reset OT-2 (home + drop tip)")
            return

        from BaituOT2Battleship.OT2_Ctrl import OT2_functions as ot

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

        Uses calibrated RGB prototypes (from calibration.json) if available.
        Always uses nearest-prototype (no L2 rejection) so we never get
        "unknown" — a low-confidence label is still better than no label.
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

        # Use calibrated prototypes or defaults
        if self._ship_rgb is not None:
            ship_rgb = self._ship_rgb
            water_rgb = self._water_rgb
        else:
            # No calibration — warn on first call
            if not hasattr(self, "_color_warned"):
                log.warning(
                    "No calibrated colour prototypes! Using defaults. "
                    "Run: python calibrate_geometry.py annotate <photo> "
                    "and use --geometry_path calibration.json"
                )
                self._color_warned = True
            from plate.battleship_plate_readout import SHIP_LIQUID_RGB, WATER_LIQUID_RGB
            ship_rgb = SHIP_LIQUID_RGB
            water_rgb = WATER_LIQUID_RGB

        # Nearest-prototype: always produce a label, never "unknown"
        dist_ship = float(np.linalg.norm(mean_rgb - ship_rgb))
        dist_water = float(np.linalg.norm(mean_rgb - water_rgb))

        if dist_ship <= dist_water:
            label = "ship"
            conf = max(0.0, min(1.0, dist_water / (dist_ship + dist_water + 1e-9)))
        else:
            label = "water"
            conf = max(0.0, min(1.0, dist_ship / (dist_ship + dist_water + 1e-9)))

        rgb = (int(mean_rgb[0]), int(mean_rgb[1]), int(mean_rgb[2]))

        log.info(
            "    %s RGB=(%d,%d,%d) dist_hit=%.1f dist_miss=%.1f → %s (%.2f)",
            _rc_to_well(row, col), *rgb, dist_ship, dist_water, label, conf,
        )

        return label, conf, rgb

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
                    from BaituOT2Battleship.OT2_Ctrl import OT2_functions as ot
                    ot.ROBOT_IP = self.cfg.robot_ip
                    ot.reconnect_last_run()
                    # Recover labware IDs from the existing run
                    self._tiprack_id = ot.get_labware_id_by_slot(TIPRACK_SLOT)
                    self._plate_id = ot.get_labware_id_by_slot(PLATE_SLOT)
                    self._indicator_id = ot.get_labware_id_by_slot(INDICATOR_SLOT)
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

    from BaituOT2Battleship.OT2_Ctrl import OT2_functions as ot
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
