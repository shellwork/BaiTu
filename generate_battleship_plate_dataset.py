"""
Generate a synthetic 96-well image dataset from Battleship layouts.

Real-world convention in this project
-------------------------------------
- Full plate layout is always 8×12.
- First 10 columns are the active experiment area.
- Last 2 columns are reserved as unused / colour-control wells.

Outputs
-------
- ``images/*.png``               synthetic plate photographs
- ``labels_full/*.npy``          full 8×12 labels (1 ship, 0 water, -1 reserved)
- ``labels_active/*.npy``        active 8×10 labels from Battleship
- ``well_mean_bgr/*.npy``        per-well mean BGR, shape (8, 12, 3)
- ``metadata.jsonl``             one JSON object per sample
- ``geometry.json``              fixed ROI geometry for RGB sampling
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np

from battleship_env import BattleshipBoard
from battleship_plate_simulation import (
    ACTIVE_COLS,
    board_grid_to_plate_labels,
    get_fixed_well_geometry,
    sample_all_wells_mean_bgr,
    simulate_photo_from_board,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic 96-well plate images from Battleship boards."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("simulated_battleship_plate_dataset"),
        help="Dataset output directory.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of synthetic plates to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed. Sample i uses seed + i.",
    )
    parser.add_argument(
        "--ship_sizes",
        type=str,
        default="5,4,3,3,2",
        help="Comma-separated Battleship ship sizes for the active 8x10 area.",
    )
    parser.add_argument(
        "--image_format",
        type=str,
        default="png",
        choices=["png", "jpg"],
        help="Image file format.",
    )
    return parser.parse_args()


def parse_ship_sizes(text: str) -> List[int]:
    sizes = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not sizes:
        raise ValueError("ship_sizes cannot be empty")
    return sizes


def ensure_dirs(root: Path) -> dict:
    subdirs = {
        "images": root / "images",
        "labels_full": root / "labels_full",
        "labels_active": root / "labels_active",
        "well_mean_bgr": root / "well_mean_bgr",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def main():
    args = parse_args()
    ship_sizes = parse_ship_sizes(args.ship_sizes)
    dirs = ensure_dirs(args.output_dir)

    geometry = get_fixed_well_geometry()
    geometry_path = args.output_dir / "geometry.json"
    geometry_path.write_text(json.dumps(geometry, indent=2))

    metadata_path = args.output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as meta_f:
        for index in range(args.n_samples):
            seed = args.seed + index
            sample_id = f"sample_{index:05d}"

            board = BattleshipBoard(
                rows=8,
                cols=ACTIVE_COLS,
                ship_sizes=ship_sizes,
                seed=seed,
            )
            active_labels = board.grid.astype(np.int8)
            full_labels = board_grid_to_plate_labels(board).astype(np.int8)
            image_bgr = simulate_photo_from_board(board, seed=seed)
            well_mean_bgr = sample_all_wells_mean_bgr(image_bgr, geometry).astype(
                np.uint8
            )

            image_path = dirs["images"] / f"{sample_id}.{args.image_format}"
            labels_full_path = dirs["labels_full"] / f"{sample_id}.npy"
            labels_active_path = dirs["labels_active"] / f"{sample_id}.npy"
            mean_bgr_path = dirs["well_mean_bgr"] / f"{sample_id}.npy"

            ok = cv2.imwrite(str(image_path), image_bgr)
            if not ok:
                raise RuntimeError(f"Failed to write image: {image_path}")
            np.save(labels_full_path, full_labels)
            np.save(labels_active_path, active_labels)
            np.save(mean_bgr_path, well_mean_bgr)

            metadata = {
                "sample_id": sample_id,
                "seed": seed,
                "image_path": str(image_path.relative_to(args.output_dir)),
                "labels_full_path": str(labels_full_path.relative_to(args.output_dir)),
                "labels_active_path": str(
                    labels_active_path.relative_to(args.output_dir)
                ),
                "well_mean_bgr_path": str(mean_bgr_path.relative_to(args.output_dir)),
                "plate_shape": [8, 12],
                "active_shape": [8, ACTIVE_COLS],
                "reserved_control_columns": [ACTIVE_COLS, ACTIVE_COLS + 1],
                "ship_sizes": ship_sizes,
                "n_ship_wells": int(active_labels.sum()),
                "ship_positions": [
                    [int(r), int(c)] for r, c in np.argwhere(active_labels == 1)
                ],
            }
            meta_f.write(json.dumps(metadata) + "\n")

    print(f"Saved dataset to: {args.output_dir}")
    print(f"Samples: {args.n_samples}")
    print(f"Active area: 8x{ACTIVE_COLS}")
    print("Reserved control columns: 10 and 11 (0-based indexing)")


if __name__ == "__main__":
    main()
