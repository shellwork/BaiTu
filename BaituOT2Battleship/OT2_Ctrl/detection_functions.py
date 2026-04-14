"""
Vision Detection Functions Module

Camera capture and vision utilities for OT-2 + camera + YOLO-based
tip detection and liquid-level checks.
"""

import os
import sys
import statistics
from datetime import datetime
from typing import Any, Dict, Optional, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


import cv2

# Import Helper from same package
from . import Helper

# detection_functions.py (Add this to the end of the file)

import matplotlib.pyplot as plt
from PIL import Image as PILImage

def Predict(
    ot_module: Any,
    model: Any,
    run_id: str,
    check_type: str,
    imaging_labware_id: str,
    imaging_well: str,
    imaging_offset: tuple,
    base_dir: str,
    step_name: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Encapsulates the entire vision workflow: Move -> Capture -> Predict -> Display.

    Args:
        ot_module: The OT2_functions module (passed dynamically to avoid circular imports).
        model: The loaded YOLO model.
        run_id: Current Run ID.
        check_type: 'pickup' or 'transfer' (determines which check to run).
        imaging_labware_id: ID of the labware used for imaging.
        imaging_well: Well to hover over.
        imaging_offset: (x, y, z) offset.
        base_dir: Directory to save images.
        step_name: Name prefix for the image file.
        **kwargs: Extra arguments for specific checks (e.g., expected_vol, expected_tips).

    Returns:
        The result dictionary from the vision check.
    """
    
    # 1. Move to Imaging Position
    print(f"[VISION] Moving to imaging position: {imaging_well}...")
    ot_module.move(
        labware_id=imaging_labware_id, 
        wellname=imaging_well, 
        offset=imaging_offset
    )
    
    # 2. Capture Image
    print("[VISION] Capturing image...")
    img_path = capture_image_with_run_id(
        run_id=run_id, 
        step_name=step_name, 
        base_dir=base_dir
    )
    
    # 3. Run Specific Model Check
    result = {}
    if img_path is not None:
        if check_type == 'pickup':
            result = check_tip_with_model(
                model=model,
                image_path=img_path,
                conf_threshold=kwargs.get('conf', 0.6),
                expected_tips=kwargs.get('expected_tips', 8)
            )
        elif check_type == 'transfer':
            result = check_liquid_level(
                model=model,
                image_path=img_path,
                conf_threshold=kwargs.get('conf', 0.6),
                expected_vol=kwargs.get('volume', 0),
                expected_tips=kwargs.get('expected_tips', 8)
            )
        else:
            raise ValueError(f"Unknown check_type: {check_type}")

        # 4. Display Results (Original + Prediction Side-by-Side)
        pred_img_path = result.get('prediction_image_path')
        if pred_img_path is not None:
            _display_comparison(img_path, pred_img_path)
    else:
        print("[VISION ERROR] No image was captured, skipping model check and display.")
    return result


def _display_comparison(original_path: str, prediction_path: str):
    """
    Helper to display original and annotated images using Matplotlib.
    """
    try:
        if not original_path or not prediction_path:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show Original
        if os.path.exists(original_path):
            axes[0].imshow(PILImage.open(original_path))
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
        # Show Prediction (YOLO Annotated)
        if os.path.exists(prediction_path):
            axes[1].imshow(PILImage.open(prediction_path))
            axes[1].set_title("YOLO Prediction Result")
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        print("[VISION] Images displayed successfully.")
        
    except Exception as e:
        print(f"[VISION WARNING] Could not display images (headless environment?): {e}")

def capture_image_for_detection(project_name: str = "tip_check") -> Optional[str]:
    """
    Open camera, capture one image using Helper utilities, then release.

    Args:
        project_name: Used to organize image names / folders.

    Returns:
        image_path or None if capture failed.
    """
    cap = Helper.initialize_camera()
    if cap is None:
        return None

    image_path = Helper.capture_live_image(cap, project_name)
    Helper.release_camera(cap)
    return image_path


def capture_image_with_run_id(
    run_id: str,
    step_name: str = "capture",
    base_dir: str = "Test",
) -> Optional[str]:
    """
    Capture image and save to:
        {as_project_dir}/{base_dir}/experiment/{run_id}/{step_name_YYYYmmdd_HHMMSS}.jpg

    Expected structure:
        - as_project/
            - OT2_Ctrl/
                - detection_functions.py
                - Helper.py
            - Test/
                - experiment/
                    - {run_id}/
                        - tip_pickup_20251124_135959.jpg
    """
    cap = Helper.initialize_camera()
    if cap is None:
        print(f"[VISION] Failed to initialize camera.")
        return None
    print(f"[VISION] Camera initialized successfully.")

    # Drop a few frames to avoid blurry first frame
    for _ in range(10):
        cap.read()

    ret, frame = cap.read()
    if not ret:
        print(f"[VISION] Failed to capture image.")
        Helper.release_camera(cap)
        return None

    # Project root is one directory above this file
    as_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"[VISION] as_project_dir: {as_project_dir}")
    # Folder: {as_project_dir}/{base_dir}/experiment/{run_id}/
    run_folder = os.path.join(as_project_dir, base_dir, "experiment", run_id)
    os.makedirs(run_folder, exist_ok=True)
    print(f"[VISION] run_folder: {run_folder}")
    # File name: {step_name}_YYYYmmdd_HHMMSS.jpg
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = f"{step_name}_{timestamp}.jpg"
    image_path = os.path.join(run_folder, image_name)
    print(f"[VISION] image_path: {image_path}")
    cv2.imwrite(image_path, frame)
    print(f"[VISION] Image saved: {image_path}")

    Helper.release_camera(cap)
    print(f"[VISION] Camera released successfully.")
    return image_path


def _save_prediction_image(image_path: str, predictions: Any) -> Optional[str]:
    """
    Save an annotated prediction image next to the original image.

    Returns:
        prediction_image_path or None if saving failed.
    """
    image_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    prediction_image_path = os.path.join(image_dir, f"{base_name}_prediction.jpg")

    try:
        if hasattr(predictions, "_images_prediction_lst"):
            pred_list = predictions._images_prediction_lst
        else:
            pred_list = [predictions]

        if not pred_list:
            print("[VISION WARNING] No predictions to save")
            return None

        pred_obj = pred_list[0]
        if not hasattr(pred_obj, "draw"):
            print("[VISION WARNING] Prediction object has no draw() method")
            return None

        annotated_image = pred_obj.draw()
        cv2.imwrite(prediction_image_path, annotated_image)
        print(f"[VISION] Prediction image saved: {prediction_image_path}")
        return prediction_image_path

    except Exception as exc:
        print(f"[VISION WARNING] Failed to save prediction image: {exc}")
        print(f"[VISION WARNING] Error type: {type(exc).__name__}")
        return None

def build_regression_function(csv_path, degree=3):
    """
    Build a regression function: input volume -> output expected height (single value).
    Uses polynomial regression on the mean channel height.
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Compute average height across channels
    channel_cols = ["Ch1","Ch2","Ch3","Ch4","Ch5","Ch6","Ch7","Ch8"]
    df["MeanHeight"] = df[channel_cols].mean(axis=1)

    # Extract regression data
    X = df["Volume"].to_numpy().reshape(-1,1)
    y = df["MeanHeight"].to_numpy()

    # Polynomial transformation
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Fit regression model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Build prediction function
    def predict(volume_ul):
        vol = np.array([[volume_ul]])
        vol_poly = poly.transform(vol)
        return float(model.predict(vol_poly))

    return predict

def check_tip_with_model(
    model: Any,
    image_path: str,
    conf_threshold: float = 0.4,
    expected_tips: int = 8,
) -> Dict[str, Any]:
    """
    Checks if all expected tips are present.
    """
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"[VISION] Tip Check: {image_path}")

    # 1. Predict & Save
    predictions = model.predict(image_path, conf=conf_threshold)
    prediction_image_path = _save_prediction_image(image_path, predictions)

    # 2. Parse
    class_counts, _, centers, liq_heights = Helper.process_predictions(predictions)
    print(f"[VISION] Detected: {class_counts}")

    # 3. Logic
    try:
        tip_presence, missing_positions = Helper.find_missing_tips(centers, expected_tips)
        passed = all(v == 1 for v in tip_presence)
        print(f"[VISION] Presence: {tip_presence}, Missing: {missing_positions}")
    except Exception as exc:
        print(f"[VISION ERROR] Analysis failed: {exc}")
        passed = class_counts.get("Tip", 0) >= expected_tips
        tip_presence = [1] if passed else [0]
        missing_positions = [] if passed else ["unknown"]

    return {
        "passed": passed,
        "tip_presence": tip_presence,
        "missing_positions": missing_positions,
        "class_counts": class_counts,
        "centers": centers,
        "prediction_image_path": prediction_image_path,
        "liquid_height_percentages": liq_heights,
        "predictions": predictions,
    }


def check_liquid_level(
    model: Any,
    image_path: str,
    conf_threshold: float = 0.4,
    expected_tips: int = 8,
    expected_vol: float = 0.0,
    LLD_Calibration_Data_path: str = "/Users/tuomasier/Desktop/02761A/project/OT2-Computer-Vision/as_project/Test/res/LLD_Calibration_Data.csv",
) -> Dict[str, Any]:
    """
    Checks liquid levels against expected volume. 
    Passes with Warning if Tip Count is correct, levels are valid, but some liquids are missed.
    """
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Configs
    tol_percent = float(os.getenv("LLD_CHANNEL_TOLERANCE_PERCENT", "5.0"))
    
    # Calculate Expected Height
    if not os.path.exists(LLD_Calibration_Data_path):
        raise FileNotFoundError(f"Calibration data missing: {LLD_Calibration_Data_path}")
    
    predict_height = build_regression_function(LLD_Calibration_Data_path, degree=3)
    expected_height = predict_height(expected_vol)
    
    print(f"[VISION] Liquid Check: {image_path}")
    print(f"[VISION] Target: {expected_vol}uL -> {expected_height:.2f}% (Tol: {tol_percent}%)")

    # 1. Predict & Parse
    predictions = model.predict(image_path, conf=conf_threshold)
    prediction_image_path = _save_prediction_image(image_path, predictions)
    class_counts, _, centers, liq_heights = Helper.process_predictions(predictions)
    
    tip_count = class_counts.get("Tip", 0)
    detected_levels = liq_heights.get("Liquid", [])
    liquid_count = len(detected_levels)

    channel_pass_status = []
    error_msg = None
    passed = False

    # 2. Evaluate Levels
    if tip_count == 0:
        error_msg = "No tips detected."
    else:
        # Check every detected liquid against expected height
        for i, lvl in enumerate(detected_levels):
            diff = abs(lvl - expected_height)
            ok = diff <= tol_percent
            channel_pass_status.append(ok)
            print(f"[VISION] Ch{i}: {lvl:.2f}% (Diff: {diff:.2f}%) -> {'PASS' if ok else 'FAIL'}")

        # === CORE LOGIC START ===
        all_levels_ok = len(channel_pass_status) > 0 and all(channel_pass_status)
        count_mismatch = liquid_count != tip_count
        tips_correct = tip_count == expected_tips

        if not all_levels_ok:
            # Case A: Actual bad levels detected -> FAIL
            passed = False
            failed_idx = [i for i, ok in enumerate(channel_pass_status) if not ok]
            error_msg = f"Levels out of range on channels {failed_idx}. Expected {expected_height:.2f}%."

        elif count_mismatch:
            # Case B: Levels OK, but count mismatch.
            # Only PASS if Tip count is correct AND existing levels are within stricter warning tolerance (5%)
            
            # Check if levels are within the stricter 5% range
            strict_pass = all(abs(lvl - expected_height) <= tol_percent for lvl in detected_levels)

            if tips_correct and strict_pass:
                passed = True
                print(f"[VISION] WARNING: Tip count {tip_count} matches expected, but only found {liquid_count} liquids.")
                print(f"[VISION] WARNING: All detected liquids are within STRICT tolerance (+/- {tol_percent}%). Proceeding.")
            else:
                passed = False
                error_msg = (f"Liquid count mismatch ({liquid_count}/{tip_count}) and "
                             f"levels/tips did not meet strict safety criteria.")
        else:
            # Case C: Perfect Match -> PASS
            passed = True
            print("[VISION] Success: All levels valid and counts match.")
        # === CORE LOGIC END ===

    if error_msg:
        print(f"[VISION ERROR] {error_msg}")

    return {
        "passed": passed,
        "detected_levels": detected_levels,
        "channel_pass_status": channel_pass_status,
        "class_counts": class_counts,
        "centers": centers,
        "prediction_image_path": prediction_image_path,
        "predictions": predictions,
        "expected_vol": expected_vol,
        "expected_height_percent": expected_height,
        "expected_tips": expected_tips,
        "tip_count": tip_count,
        "liquid_count": liquid_count,
        "error_msg": error_msg,
    }

if __name__ == "__main__":
    print("Vision Detection Functions Module")
    print("Use this module with OT2_functions and Protocol/ptc.py for tip and liquid checks.")
