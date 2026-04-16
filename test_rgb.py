"""
Test RGB recognition on a plate image.

Usage:
  # Use calibration.json (geometry + colour prototypes)
  python test_rgb.py <image_path> --calibration calibration.json

  # Use geometry.json with default colour prototypes
  python test_rgb.py <image_path> --calibration geometry.json

  # Capture a new photo and test immediately
  python test_rgb.py capture --calibration calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np

from plate.battleship_plate_readout import (
    classify_ship_water_fixed_rgb_tolerance,
    classify_ship_water_from_mean_bgr,
    mean_bgr_to_mean_rgb,
    sample_well_mean_bgr,
)

ROWS = 8
COLS = 10  # active area only
ROW_LABELS = [chr(ord("A") + i) for i in range(ROWS)]


def capture_photo() -> str:
    from datetime import datetime

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        sys.exit(1)
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Failed to capture")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"test_rgb_{ts}.jpg"
    cv2.imwrite(path, frame)
    print(f"Image captured: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Test RGB well recognition")
    parser.add_argument("image", help="image path, or 'capture' to take a new photo")
    parser.add_argument("--calibration", required=True, help="calibration.json or geometry.json")
    args = parser.parse_args()

    # Load calibration
    with open(args.calibration) as f:
        calib = json.load(f)

    geometry = {
        "rows": calib["rows"],
        "cols": calib["cols"],
        "row_centers": calib["row_centers"],
        "col_centers": calib["col_centers"],
        "well_radius": calib["well_radius"],
    }

    # Colour prototypes (from calibration or defaults)
    if "ship_rgb" in calib:
        ship_rgb = np.array(calib["ship_rgb"], dtype=np.float32)
        water_rgb = np.array(calib["water_rgb"], dtype=np.float32)
        l2_tol = calib.get("rgb_l2_tolerance", 48.0)
        print(f"Colour prototypes (from calibration):")
    else:
        ship_rgb = np.array([166.0, 92.0, 135.0], dtype=np.float32)
        water_rgb = np.array([104.0, 91.0, 134.0], dtype=np.float32)
        l2_tol = 48.0
        print(f"Colour prototypes (defaults):")
    print(f"  HIT  (ship)  RGB: {ship_rgb.tolist()}")
    print(f"  MISS (water) RGB: {water_rgb.tolist()}")
    print(f"  L2 tolerance:     {l2_tol}")
    print()

    # Get image
    if args.image == "capture":
        image_path = capture_photo()
    else:
        image_path = args.image

    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot read {image_path}")
        sys.exit(1)
    print(f"Image: {image_path}  (shape: {img.shape})")
    print()

    # Classify all active wells
    print(f"{'Well':<6} {'Label':<8} {'Conf':>5}   {'RGB (measured)':>20}   {'dist_hit':>8} {'dist_miss':>9}")
    print("-" * 70)

    hit_count = 0
    miss_count = 0
    unknown_count = 0

    for r in range(ROWS):
        for c in range(COLS):
            well = f"{ROW_LABELS[r]}{c+1}"
            mean_bgr = sample_well_mean_bgr(img, r, c, geometry)
            mean_rgb_vec = mean_bgr_to_mean_rgb(mean_bgr)

            label, conf = classify_ship_water_fixed_rgb_tolerance(
                mean_bgr,
                ship_rgb=ship_rgb,
                water_rgb=water_rgb,
                l2_max=l2_tol,
            )

            # If unknown, use nearest-prototype fallback
            if label == "unknown":
                ship_bgr = np.array([ship_rgb[2], ship_rgb[1], ship_rgb[0]], dtype=np.float32)
                water_bgr = np.array([water_rgb[2], water_rgb[1], water_rgb[0]], dtype=np.float32)
                label, conf = classify_ship_water_from_mean_bgr(
                    mean_bgr, red_prototype_bgr=ship_bgr, blue_prototype_bgr=water_bgr,
                )
                label = label + "*"  # mark as fallback

            dist_hit = float(np.linalg.norm(mean_rgb_vec - ship_rgb))
            dist_miss = float(np.linalg.norm(mean_rgb_vec - water_rgb))
            rgb_str = f"({mean_rgb_vec[0]:.0f},{mean_rgb_vec[1]:.0f},{mean_rgb_vec[2]:.0f})"

            marker = "<<<" if "ship" in label else ""
            print(f"{well:<6} {label:<8} {conf:>5.2f}   {rgb_str:>20}   {dist_hit:>8.1f} {dist_miss:>9.1f}  {marker}")

            if "ship" in label:
                hit_count += 1
            elif "water" in label:
                miss_count += 1
            else:
                unknown_count += 1

    print("-" * 70)
    print(f"Total: {hit_count} HIT,  {miss_count} MISS,  {unknown_count} UNKNOWN  (of {ROWS*COLS} wells)")

    # Print board-like view
    print(f"\nBoard view (X=HIT, O=MISS):")
    print(f"     " + "  ".join(f"{c+1:>2}" for c in range(COLS)))
    for r in range(ROWS):
        row_str = f"  {ROW_LABELS[r]}  "
        for c in range(COLS):
            mean_bgr = sample_well_mean_bgr(img, r, c, geometry)
            label, _ = classify_ship_water_fixed_rgb_tolerance(
                mean_bgr, ship_rgb=ship_rgb, water_rgb=water_rgb, l2_max=l2_tol,
            )
            if label == "unknown":
                ship_bgr = np.array([ship_rgb[2], ship_rgb[1], ship_rgb[0]], dtype=np.float32)
                water_bgr = np.array([water_rgb[2], water_rgb[1], water_rgb[0]], dtype=np.float32)
                label, _ = classify_ship_water_from_mean_bgr(
                    mean_bgr, red_prototype_bgr=ship_bgr, blue_prototype_bgr=water_bgr,
                )
            sym = " X" if label == "ship" else " O"
            row_str += f" {sym}"
        print(row_str)


if __name__ == "__main__":
    main()
