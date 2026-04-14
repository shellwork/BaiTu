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
# Slot 1: tiprack
TIPRACK_SLOT = "1"
TIPRACK_LABWARE = "opentrons_96_tiprack_300ul"
# Slot 2: 96-well plate (target)
PLATE_SLOT = "2"
PLATE_LABWARE = "corning_96_wellplate_360ul_flat"
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


@dataclass
class LoopConfig:
    # Strategy
    strategy: str = "prob"
    seed: Optional[int] = None

    # OT-2 connection
    robot_ip: str = "169.254.200.128"

    # Camera
    geometry_path: Optional[str] = None  # pre-calibrated geometry JSON
    color_develop_seconds: float = 30.0

    # Output
    output_dir: str = "ot2_loop_results"
    checkpoint_path: Optional[str] = None  # resume from checkpoint

    # Debug
    dry_run: bool = False

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

    def _dispense_board_liquids(
        self,
        naoh_wells: List[str],
        h2o_wells: List[str],
    ) -> None:
        """OT-2 Phase 1: fill all 80 active wells with NaOH or H2O."""
        from BaituOT2Battleship.OT2_Ctrl import OT2_functions as ot

        # NaOH → empty/miss wells (one tip for all)
        tip_well = _tip_well_name(self._tip_index)
        self._tip_index += 1
        ot.pick_up(self._tiprack_id, tip_well)
        for w in naoh_wells:
            ot.aspirate(FILL_VOLUME, self._naoh_id, NAOH_SOURCE_WELL)
            ot.dispense(FILL_VOLUME, self._plate_id, w)
        ot.unload_to_trash()

        # H2O → ship/hit wells (fresh tip to avoid contamination)
        tip_well = _tip_well_name(self._tip_index)
        self._tip_index += 1
        ot.pick_up(self._tiprack_id, tip_well)
        for w in h2o_wells:
            ot.aspirate(FILL_VOLUME, self._h2o_id, H2O_SOURCE_WELL)
            ot.dispense(FILL_VOLUME, self._plate_id, w)
        ot.unload_to_trash()

        log.info("Phase 1 complete: NaOH→%d wells, H2O→%d wells", len(naoh_wells), len(h2o_wells))

    def _calibrate_geometry(self) -> Dict:
        """Get well geometry: from file or by photographing the plate."""
        if self.cfg.geometry_path:
            with open(self.cfg.geometry_path) as f:
                geo = json.load(f)
            log.info("Loaded pre-calibrated geometry from %s", self.cfg.geometry_path)
            return geo

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

    def _dispense_indicator(self, well_name: str) -> None:
        """Phase 2: dispense indicator into one well (fresh tip each time)."""
        if self.cfg.dry_run:
            log.info("[DRY RUN] Dispense indicator → %s", well_name)
            return

        from BaituOT2Battleship.OT2_Ctrl import OT2_functions as ot

        tip_well = _tip_well_name(self._tip_index)
        self._tip_index += 1

        ot.pick_up(self._tiprack_id, tip_well)
        ot.aspirate(INDICATOR_VOLUME, self._indicator_id, INDICATOR_SOURCE_WELL)
        ot.dispense(INDICATOR_VOLUME, self._plate_id, well_name)
        ot.unload_to_trash()

    def _capture_image(self, tag: str) -> str:
        """Capture a plate image. Returns the saved file path."""
        if self.cfg.dry_run:
            return self._capture_synthetic_image(tag)

        from BaituOT2Battleship.OT2_Ctrl import OT2_functions as ot
        ot.home()  # move arm out of camera view

        from BaituOT2Battleship.OT2_Ctrl.Helper import capture_hd_image_with_lock

        for attempt in range(self.cfg.max_camera_retries):
            try:
                path = capture_hd_image_with_lock(
                    project_name=str(self._images_dir / tag),
                )
                if path and os.path.exists(path):
                    return path
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

    def _classify_well(
        self,
        image_path: str,
        row: int,
        col: int,
    ) -> Tuple[str, float, Tuple[int, int, int]]:
        """Read one well colour from the plate image."""
        if self.cfg.dry_run:
            is_hit = bool(self.board.grid[row, col])
            label = "ship" if is_hit else "water"
            return label, 1.0, (0, 0, 0)

        image_bgr = load_image_bgr(image_path)
        label, mean_bgr, conf = query_well_fixed_geometry_rgb(
            image_bgr, row, col, self.geometry,
        )

        # Fallback for "unknown" using nearest-prototype
        if label == "unknown":
            label_fb, conf_fb = classify_ship_water_from_mean_bgr(mean_bgr)
            log.warning(
                "Well %s classified as unknown (rgb=%s), fallback → %s (%.2f)",
                _rc_to_well(row, col), tuple(mean_bgr.tolist()), label_fb, conf_fb,
            )
            label = label_fb
            conf = conf_fb

        rgb = tuple(int(v) for v in cv2.cvtColor(
            mean_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2RGB,
        ).flatten())

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
            self._setup_ot2()
            self._setup_board()
            self.model = Game(board_rows=BOARD_ROWS, board_cols=BOARD_COLS)

        self.geometry = self._calibrate_geometry()

        log.info("Starting game loop (strategy=%s) ...", self.cfg.strategy)

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

            # 3) Wait for colour development
            if not self.cfg.dry_run:
                log.info("  Waiting %.0fs for colour development ...", self.cfg.color_develop_seconds)
                time.sleep(self.cfg.color_develop_seconds)

            # 4) Capture image and classify
            image_path = self._capture_image(f"step_{self._step:03d}")
            label, conf, mean_rgb = self._classify_well(image_path, row, col)
            is_hit = (label == "ship")

            # 5) Update ground truth (get sunk info)
            _actual_hit, sunk_ship = self.board.query(row, col)

            # 6) Update model
            self.model.update(row, col, is_hit=is_hit, sunk_ship=sunk_ship)

            # 7) Update results matrix
            self.results_matrix[row, col] = 1 if is_hit else 0

            # 8) Record
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

        # Home the robot
        if not self.cfg.dry_run:
            from BaituOT2Battleship.OT2_Ctrl import OT2_functions as ot
            ot.home()

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
    parser.add_argument("--color_develop_seconds", type=float, default=30.0)
    parser.add_argument("--geometry_path", default=None)
    parser.add_argument("--output_dir", default="ot2_loop_results")
    parser.add_argument("--resume", default=None, help="checkpoint JSON path")
    parser.add_argument("--dry_run", action="store_true",
                        help="simulate without OT-2 hardware")

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
    )

    loop = OT2BattleshipLoop(config)
    loop.run()


if __name__ == "__main__":
    main()
