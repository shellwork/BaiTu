import os
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import cv2


def process_predictions(
    predictions: Any,
) -> Tuple[Dict[str, int], Dict[str, List[List[float]]],
           Dict[str, List[List[float]]], Dict[str, List[float]]]:
    """
    Parse a super-gradients style predictions object.
    
    Fixed Logic: Collects all boxes first, sorts them spatially, 
    and then matches Liquids to their corresponding Tips to calculate percentage.
    """
    # Initialize containers
    class_counts: Dict[str, int] = {"Tip": 0, "Liquid": 0}
    bounding_boxes: Dict[str, List[List[float]]] = {"Tip": [], "Liquid": []}
    bounding_box_centers: Dict[str, List[List[float]]] = {"Tip": [], "Liquid": []}
    liquid_height_percentages: Dict[str, List[float]] = {"Liquid": []}

    # Handle batch vs single prediction
    if hasattr(predictions, "_images_prediction_lst"):
        prediction_list = predictions._images_prediction_lst
    else:
        prediction_list = [predictions]

    for image_prediction in prediction_list:
        prediction = image_prediction.prediction
        labels = prediction.labels
        bboxes = prediction.bboxes_xyxy
        class_names = image_prediction.class_names

        # === Step 1: Collect Raw Data ===
        for label, bbox in zip(labels, bboxes):
            class_name = class_names[int(label)]
            
            # Ensure keys exist
            if class_name not in class_counts: class_counts[class_name] = 0
            if class_name not in bounding_boxes: bounding_boxes[class_name] = []
            if class_name not in bounding_box_centers: bounding_box_centers[class_name] = []

            # 1. Update Counts
            class_counts[class_name] += 1
            
            # 2. Store BBox
            bbox_list = bbox.tolist()
            bounding_boxes[class_name].append(bbox_list)

            # 3. Store Center
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            bounding_box_centers[class_name].append([float(x_center), float(y_center)])

        # === Step 2: Sort Spatially (Left to Right) ===
        # Sort Tips and Liquids by x_min (bbox[0])
        # This is critical for mapping the correct liquid to the correct tip
        sorted_tips = sorted(bounding_boxes.get("Tip", []), key=lambda b: b[0])
        sorted_liquids = sorted(bounding_boxes.get("Liquid", []), key=lambda b: b[0])
        
        # Also update the global bounding_box_centers to be sorted (for downstream logic)
        for cls_name in bounding_box_centers:
            bounding_box_centers[cls_name].sort(key=lambda c: c[0])

        # === Step 3: Match Liquid to Tip & Calculate Percentage ===
        # Logic: For each Liquid, find the Tip that spatially encloses it (or is closest).
        
        calculated_levels = []
        
        if not sorted_tips:
            print("[MODEL WARNING] Liquid detected but no Tips found. Cannot calculate percentage.")
        else:
            for liq_box in sorted_liquids:
                liq_x_center = (liq_box[0] + liq_box[2]) / 2
                
                # Find the corresponding Tip: The one with the minimal X-distance to the liquid
                # (Simple sorting isn't enough if a tip is missed in the middle, so we find nearest neighbor)
                closest_tip = min(
                    sorted_tips, 
                    key=lambda t_box: abs(((t_box[0] + t_box[2]) / 2) - liq_x_center)
                )
                
                # Calculate Heights
                liquid_h = float(liq_box[3] - liq_box[1])
                tip_h = float(closest_tip[3] - closest_tip[1])
                
                if tip_h > 0:
                    pct = (liquid_h / tip_h) * 100.0
                    calculated_levels.append(pct)
                else:
                    calculated_levels.append(0.0)

        liquid_height_percentages["Liquid"] = calculated_levels
        print(f"[MODEL] Liquid Level Percentages (Sorted): {calculated_levels}")

    return class_counts, bounding_boxes, bounding_box_centers, liquid_height_percentages

def find_missing_tips(
    bounding_box_centers: Dict[str, List[List[float]]],
    expected_tip_count: int = 8,
) -> Tuple[List[int], List[int]]:
    """
    Infer missing tips based on x-coordinate spacing.

    Args:
        bounding_box_centers: class_name -> list of [x_center, y_center].
        expected_tip_count: number of channels on the multichannel pipette.

    Returns:
        tip_presence: list of 0/1, length = expected_tip_count.
        missing_tip_positions: 1-based indices of missing tips.
    """
    tip_centers = bounding_box_centers.get("Tip", [])
    tip_presence = [1] * expected_tip_count
    missing_tip_positions: List[int] = []

    # No tips detected at all
    if not tip_centers:
        return [0] * expected_tip_count, list(range(1, expected_tip_count + 1))

    # Exact count -> assume no missing tips
    if len(tip_centers) == expected_tip_count:
        return tip_presence, missing_tip_positions

    # Estimate horizontal spacing
    expected_horizontal_distance = (tip_centers[-1][0] - tip_centers[0][0]) / max(
        expected_tip_count - 1, 1
    )

    for i in range(expected_tip_count):
        expected_center_x = tip_centers[0][0] + i * expected_horizontal_distance
        found = False
        for detected_center in tip_centers:
            if abs(detected_center[0] - expected_center_x) <= expected_horizontal_distance / 2:
                found = True
                break

        if not found:
            tip_presence[i] = 0
            missing_tip_positions.append(i + 1)

    return tip_presence, missing_tip_positions


def create_project(project_name: str = "default_project") -> str:
    """
    Create a project folder (if needed) and return a unique image path.
    Example:
        <cwd>/<project_name>/<project_name>_<timestamp>_hd.jpg
    """
    project_folder = os.path.join(os.path.abspath("."), project_name)
    os.makedirs(project_folder, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    image_name = f"{project_name}_{current_time}_hd.jpg"
    return os.path.join(project_folder, image_name)


def initialize_camera(camera_id: int = 0) -> Optional[cv2.VideoCapture]:
    """
    Initialize a camera device and set resolution.

    Returns:
        OpenCV VideoCapture object or None if opening failed.
    """
    cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"Failed to open camera {camera_id}")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def capture_live_image(
    cap: Optional[cv2.VideoCapture],
    project_name: str,
) -> Optional[str]:
    """
    Capture a single frame from an open camera and save it.

    Returns:
        image_path or None if capture failed.
    """
    if cap is None:
        print("capture_live_image called with no active camera")
        return None

    # Drop several frames to stabilize exposure
    for _ in range(10):
        cap.read()

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        return None

    image_path = create_project(project_name)
    cv2.imwrite(image_path, frame)
    print(f"Image captured: {image_path}")
    return image_path


def release_camera(cap: Optional[cv2.VideoCapture]) -> None:
    """Release the camera resource if opened."""
    if cap is not None:
        cap.release()
        print("Camera released")


def capture_hd_image_with_lock(project_name: str = "default_project") -> Optional[str]:
    """
    One-shot helper: open camera, capture one image, then release it.

    Returns:
        image_path or None if capture failed.
    """
    cap = initialize_camera()
    if cap is None:
        return None

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        release_camera(cap)
        return None

    image_path = create_project(project_name)
    cv2.imwrite(image_path, frame)
    print(f"HD image captured: {image_path}")
    release_camera(cap)
    return image_path


def save_predictions(
    predictions: Any,
    output_folder: str = "output_predictions",
) -> Optional[str]:
    """
    Save prediction visualization and rename with timestamp.

    Returns:
        Full path to saved prediction image, or None on failure.
    """
    os.makedirs(output_folder, exist_ok=True)

    try:
        predictions.save(output_folder=output_folder)
    except Exception as exc:
        print(f"Failed to save predictions: {exc}")
        return None

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    original_file_path = os.path.join(output_folder, "pred_0.jpg")
    new_file_path = os.path.join(output_folder, f"pred_{timestamp}.jpg")

    if not os.path.exists(original_file_path):
        print(f"Prediction image not found at {original_file_path}")
        return None

    os.rename(original_file_path, new_file_path)
    return new_file_path
