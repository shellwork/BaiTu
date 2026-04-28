"""
Manual geometry + colour calibration tool.

Usage (run from repository root):
  1. Take a photo:
       python -m hardware.calibrate_geometry capture

  2. Calibrate geometry (click 4 corners) + colour (click hit/miss wells):
       python -m hardware.calibrate_geometry annotate <image_path>

  Output: hardware/calibration.json  (geometry + colour prototypes)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROWS = 8
COLS = 12

# ═══════════════════════════════════════════════════════════════════════
# Shared state for click callbacks
# ═══════════════════════════════════════════════════════════════════════

clicks: list = []
img_display = None
click_limit = 0
window_name = ""


def _on_click(event, x, y, flags, param):
    global img_display
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if len(clicks) >= click_limit:
        return

    clicks.append((x, y))
    idx = len(clicks)
    cv2.circle(img_display, (x, y), 6, (0, 255, 0), 2)
    cv2.putText(img_display, str(idx), (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow(window_name, img_display)


def collect_clicks(
    img,
    title: str,
    labels: list[str],
    min_clicks: int = 2,
) -> list[tuple[int, int]]:
    """Show image, collect up to ``len(labels)`` clicks, return list of (x, y).

    Press 'd' once you have enough samples (>= ``min_clicks``) to finish
    early; 'r' to reset; 'q' to abort.
    """
    global clicks, img_display, click_limit, window_name

    clicks = []
    img_display = img.copy()
    click_limit = len(labels)
    window_name = title

    print(f"\n{title}")
    print(f"Click between {min_clicks} and {click_limit} wells:")
    for i, label in enumerate(labels):
        print(f"  {i+1}. {label}")
    print("Press 'd' when done, 'r' to reset, 'q' to abort.\n")

    cv2.imshow(window_name, img_display)
    cv2.setMouseCallback(window_name, _on_click)

    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return []
        if key == ord("r"):
            clicks = []
            img_display = img.copy()
            cv2.imshow(window_name, img_display)
            print("  Reset — click again.")
        if key == ord("d") and len(clicks) >= min_clicks:
            break
        if len(clicks) == click_limit:
            break

    cv2.destroyAllWindows()

    for i, (x, y) in enumerate(clicks):
        lbl = labels[i] if i < len(labels) else f"sample #{i+1}"
        print(f"  [{i+1}] {lbl} = ({x}, {y})")

    return list(clicks)


# ═══════════════════════════════════════════════════════════════════════
# Geometry computation
# ═══════════════════════════════════════════════════════════════════════

def compute_geometry(corners: list, well_radius: int) -> dict:
    (x_a1, y_a1), (x_a12, y_a12), (x_h1, y_h1), (x_h12, y_h12) = corners

    row_centers = []
    col_centers = []

    for r in range(ROWS):
        t = r / (ROWS - 1)
        y_left = y_a1 + t * (y_h1 - y_a1)
        y_right = y_a12 + t * (y_h12 - y_a12)
        row_centers.append(int(round((y_left + y_right) / 2)))

    for c in range(COLS):
        t = c / (COLS - 1)
        x_top = x_a1 + t * (x_a12 - x_a1)
        x_bot = x_h1 + t * (x_h12 - x_h1)
        col_centers.append(int(round((x_top + x_bot) / 2)))

    return {
        "rows": ROWS,
        "cols": COLS,
        "row_centers": row_centers,
        "col_centers": col_centers,
        "well_radius": well_radius,
    }


def draw_grid(img, geo):
    vis = img.copy()
    r = geo["well_radius"]
    for ri, cy in enumerate(geo["row_centers"]):
        for ci, cx in enumerate(geo["col_centers"]):
            cv2.circle(vis, (cx, cy), r, (0, 255, 0), 1)
            cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)
            label = f"{chr(65+ri)}{ci+1}"
            cv2.putText(vis, label, (cx - 8, cy - r - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0), 1)
    return vis


# ═══════════════════════════════════════════════════════════════════════
# Colour sampling
# ═══════════════════════════════════════════════════════════════════════

def sample_rgb_at(img_bgr, x, y, radius=8) -> np.ndarray:
    """Sample mean BGR in a small disk around (x, y)."""
    h, w = img_bgr.shape[:2]
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    patch = img_bgr[y0:y1, x0:x1]
    ph, pw = patch.shape[:2]
    yy, xx = np.mgrid[0:ph, 0:pw]
    dist = np.sqrt((xx - (x - x0))**2 + (yy - (y - y0))**2)
    pixels = patch[dist < radius]
    if len(pixels) == 0:
        return np.zeros(3, dtype=np.float32)
    mean_bgr = pixels.mean(axis=0).astype(np.float32)
    # Convert BGR → RGB
    return np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float32)


def sample_lab_at(img_bgr, x, y, radius=8) -> np.ndarray:
    """Sample mean Lab (OpenCV 8-bit) in a small disk around (x, y)."""
    h, w = img_bgr.shape[:2]
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    patch = img_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros(3, dtype=np.float32)
    lab_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    ph, pw = lab_patch.shape[:2]
    yy, xx = np.mgrid[0:ph, 0:pw]
    dist = np.sqrt((xx - (x - x0))**2 + (yy - (y - y0))**2)
    pixels = lab_patch[dist < radius]
    if len(pixels) == 0:
        return np.zeros(3, dtype=np.float32)
    return pixels.mean(axis=0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════════

def cmd_capture():
    from datetime import datetime

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Failed to capture")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"plate_photo_{ts}.jpg"
    cv2.imwrite(path, frame)
    print(f"Image saved: {path}  (shape: {frame.shape})")
    print(f"Next step:   python -m hardware.calibrate_geometry annotate {path}")


def cmd_annotate(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot read {image_path}")
        return

    print(f"Image loaded: {image_path}  (shape: {img.shape})")

    # ── Step 1: Geometry — click 4 corners ──────────────────────────
    corner_labels = [
        "A1 (top-left)", "A12 (top-right)",
        "H1 (bottom-left)", "H12 (bottom-right)",
    ]
    corners = collect_clicks(img, "STEP 1/3: Click 4 corner wells", corner_labels)
    if len(corners) != 4:
        return

    (x_a1, _), (x_a12, _), _, _ = corners
    col_spacing = abs(x_a12 - x_a1) / (COLS - 1)
    well_radius = max(5, int(col_spacing * 0.4))
    geo = compute_geometry(corners, well_radius)

    # Show grid for verification
    vis = draw_grid(img, geo)
    cv2.imshow("Verify grid — press any key to continue, 'q' to cancel", vis)
    print(f"\nWell radius: {well_radius} px")
    print("Verify green circles align with wells. Press any key to continue.")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    if key == ord("q"):
        print("Cancelled.")
        return

    # ── Step 2: Colour — click hit (ship) wells ────────────────────
    # Take up to 6 samples per class — more samples → tighter prototype.
    hit_clicks = collect_clicks(
        vis, "STEP 2/3: Click 2-6 HIT wells (ship / H2O / no colour change)",
        [f"hit well #{i+1}" for i in range(6)],
    )
    if not hit_clicks:
        return

    hit_rgbs = [sample_rgb_at(img, x, y, well_radius) for x, y in hit_clicks]
    hit_labs = [sample_lab_at(img, x, y, well_radius) for x, y in hit_clicks]
    hit_proto = np.mean(hit_rgbs, axis=0).astype(np.float32)
    hit_lab_proto = np.mean(hit_labs, axis=0).astype(np.float32)
    print(f"\n  HIT prototype RGB: [{hit_proto[0]:.0f}, {hit_proto[1]:.0f}, {hit_proto[2]:.0f}]")
    print(f"  HIT prototype Lab: [{hit_lab_proto[0]:.1f}, {hit_lab_proto[1]:.1f}, {hit_lab_proto[2]:.1f}]")
    for i, (rgb, lab) in enumerate(zip(hit_rgbs, hit_labs)):
        print(f"    sample {i+1}: RGB=[{rgb[0]:.0f},{rgb[1]:.0f},{rgb[2]:.0f}] "
              f"Lab=[{lab[0]:.1f},{lab[1]:.1f},{lab[2]:.1f}]")

    # ── Step 3: Colour — click miss (water/NaOH) wells ─────────────
    miss_clicks = collect_clicks(
        vis, "STEP 3/3: Click 2-6 MISS wells (NaOH / colour changed)",
        [f"miss well #{i+1}" for i in range(6)],
    )
    if not miss_clicks:
        return

    miss_rgbs = [sample_rgb_at(img, x, y, well_radius) for x, y in miss_clicks]
    miss_labs = [sample_lab_at(img, x, y, well_radius) for x, y in miss_clicks]
    miss_proto = np.mean(miss_rgbs, axis=0).astype(np.float32)
    miss_lab_proto = np.mean(miss_labs, axis=0).astype(np.float32)
    print(f"\n  MISS prototype RGB: [{miss_proto[0]:.0f}, {miss_proto[1]:.0f}, {miss_proto[2]:.0f}]")
    print(f"  MISS prototype Lab: [{miss_lab_proto[0]:.1f}, {miss_lab_proto[1]:.1f}, {miss_lab_proto[2]:.1f}]")
    for i, (rgb, lab) in enumerate(zip(miss_rgbs, miss_labs)):
        print(f"    sample {i+1}: RGB=[{rgb[0]:.0f},{rgb[1]:.0f},{rgb[2]:.0f}] "
              f"Lab=[{lab[0]:.1f},{lab[1]:.1f},{lab[2]:.1f}]")

    # Compute recommended tolerance and Lab-space discriminant quality
    proto_dist = float(np.linalg.norm(hit_proto - miss_proto))
    tolerance = max(20.0, proto_dist * 0.6)
    lab_proto_dist = float(np.linalg.norm(hit_lab_proto - miss_lab_proto))

    # Project all calibration samples onto the Lab discriminant axis to
    # report a quick quality metric (margin / class spread).
    diff = hit_lab_proto - miss_lab_proto
    direction = diff / (lab_proto_dist + 1e-9)
    threshold = float(((hit_lab_proto + miss_lab_proto) / 2.0) @ direction)
    hit_proj = np.array([float(lab @ direction) - threshold for lab in hit_labs])
    miss_proj = np.array([float(lab @ direction) - threshold for lab in miss_labs])
    inter_class = (float(hit_proj.mean()) - float(miss_proj.mean()))
    intra_class = float(hit_proj.std() + miss_proj.std()) or 1e-6
    fisher = inter_class / intra_class

    print(f"\n  RGB prototype distance: {proto_dist:.1f}  (tolerance → {tolerance:.1f})")
    print(f"  Lab prototype distance: {lab_proto_dist:.1f}")
    print(f"  Lab Fisher-style score: {fisher:+.2f}  "
          f"(>= 3.0 is comfortable, < 1.5 means classes overlap badly)")

    # ── Save calibration.json ───────────────────────────────────────
    calib = {
        **geo,
        "ship_rgb": [float(v) for v in hit_proto],
        "water_rgb": [float(v) for v in miss_proto],
        "rgb_l2_tolerance": round(tolerance, 1),
        # Preferred LAB-space discriminant — used by the OT-2 loop when present.
        "lab_ship": [float(v) for v in hit_lab_proto],
        "lab_water": [float(v) for v in miss_lab_proto],
        "lab_proto_distance": round(lab_proto_dist, 2),
        "lab_fisher_score": round(float(fisher), 2),
    }

    # Always write calibration.json next to this script (hardware/).
    out_path = str(Path(__file__).parent / "calibration.json")
    with open(out_path, "w") as f:
        json.dump(calib, f, indent=2)

    print(f"\n{'='*55}")
    print(f"Saved: {out_path}")
    print(f"  Geometry:  {geo['rows']}x{geo['cols']} grid, radius={geo['well_radius']}px")
    print(f"  HIT  RGB:  {calib['ship_rgb']}")
    print(f"  MISS RGB:  {calib['water_rgb']}")
    print(f"  Tolerance: {calib['rgb_l2_tolerance']}")
    print(f"{'='*55}")

    # Save annotated image
    vis_path = Path(image_path).stem + "_calibrated.jpg"
    cv2.imwrite(vis_path, vis)
    print(f"  preview: {vis_path}")

    print(f"\nUsage:")
    print(f"  python -m hardware.battleship_ot2_loop --geometry_path {out_path} --strategy prob --seed 42")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "capture":
        cmd_capture()
    elif cmd == "annotate":
        if len(sys.argv) < 3:
            print("Usage: python -m hardware.calibrate_geometry annotate <image_path>")
            sys.exit(1)
        cmd_annotate(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")
        print("Commands: capture, annotate")
