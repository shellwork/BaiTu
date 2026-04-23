import matplotlib.pyplot as plt
from PIL import Image as PILImage
from OT2_Ctrl import Helper

print("="*60)
print("CAMERA IMAGE CAPTURE TEST")
print("="*60)

try:
    # Step 1: Initialize camera
    print("\n[1] Initializing camera...")
    cap = Helper.initialize_camera()
    
    if cap is None:
        print("✗ Failed to initialize camera. Check if camera is connected.")
    else:
        print("✓ Camera initialized successfully")
        
        # Step 2: Capture image using Helper function
        # This automatically warms up camera and saves to project folder
        print("\n[2] Capturing image from camera...")
        image_path = Helper.capture_live_image(cap, project_name="camera_test")
        
        if image_path:
            print(f"✓ Image captured and saved: {image_path}")
            
            # Step 2.5: Display the captured image
            print("\n[3] Displaying image...")
            try:
                img = PILImage.open(image_path)
                plt.figure(figsize=(8, 6))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Captured Image: {Path(image_path).name}")
                plt.tight_layout()
                plt.show()
                print("✓ Image displayed successfully")
            except Exception as display_error:
                print(f"⚠ Could not display image: {display_error}")
        else:
            print("✗ Failed to capture image from camera")
        
        # Step 3: Release camera
        print("\n[4] Releasing camera...")
        Helper.release_camera(cap)
        print("✓ Camera released")
        
except Exception as e:
    print(f"✗ Error during camera test: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✓ CAMERA TEST COMPLETE")
print("="*60)


import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#%matplotlib inline

ROWS, COLS = 8, 12

def cluster_1d(values, k):
    """Collapse detected circle coordinates into exactly k cluster centres."""
    centers = sorted(set(np.round(values).astype(int).tolist()))
    while len(centers) > k:
        diffs = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        i_min = int(np.argmin(diffs))
        merged = (centers[i_min] + centers[i_min + 1]) // 2
        centers = centers[:i_min] + [merged] + centers[i_min + 2:]
    return centers if len(centers) == k else None

def hough_circles_adaptive(image, expected_radius):
    """Try Hough circle detection with given expected radius."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), sigmaX=2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(expected_radius * 1.8),
        param1=50,
        param2=28,
        minRadius=int(expected_radius * 0.65),
        maxRadius=int(expected_radius * 1.45),
    )
    return circles[0] if circles is not None else None

def detect_wells_adaptive(image_path):
    """
    Detect 96-well plate using adaptive Hough circle detection.
    Tries multiple expected radius values to find the best match.
    """
    if isinstance(image_path, str):
        image_path = Path(image_path)
    
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"✗ Failed to read image: {image_path}")
        return None, img
    
    print(f"✓ Image loaded: {image_path.name}")
    print(f"  Dimensions: {img.shape}")
    
    # Try multiple expected radii (wells can vary based on camera distance/resolution)
    radius_attempts = [28, 35, 42, 50, 22, 18, 60, 75]
    
    best_circles = None
    best_count = 0
    best_radius = 28
    
    for expected_radius in radius_attempts:
        circles = hough_circles_adaptive(img, expected_radius)
        
        if circles is not None:
            n_circles = len(circles)
            print(f"  Radius {expected_radius}: {n_circles} circles detected")
            
            # Pick the one closest to 96
            if abs(n_circles - 96) < abs(best_count - 96):
                best_circles = circles
                best_count = n_circles
                best_radius = expected_radius
    
    if best_circles is None or best_count < 48:
        print(f"\n✗ Could not detect enough circles (got {best_count}, need at least 48)")
        print("  Manual annotation may be needed for this image.")
        return None, img
    
    print(f"\n✓ Best result: {best_count} circles with expected_radius={best_radius}")
    
    # Cluster into 8x12 grid
    row_centers = cluster_1d(best_circles[:, 1], ROWS)
    col_centers = cluster_1d(best_circles[:, 0], COLS)
    
    if row_centers is None or col_centers is None:
        print("✗ Could not cluster into 8×12 grid")
        print("  Detection may need manual annotation.")
        return None, img
    
    median_radius = int(np.median(best_circles[:, 2]))
    print(f"✓ Clustered into 8×12 grid, median radius: {median_radius} px")
    
    # Extract RGB from each well
    wells_data = []
    rgb_matrix = np.zeros((ROWS, COLS), dtype=object)
    H, W = img.shape[:2]
    
    for ri, cy in enumerate(sorted(row_centers)):
        for ci, cx in enumerate(sorted(col_centers)):
            # Sample inner region (60% of radius)
            r_inner = max(1, int(median_radius * 0.60))
            
            y0, y1 = max(0, cy - median_radius), min(H, cy + median_radius + 1)
            x0, x1 = max(0, cx - median_radius), min(W, cx + median_radius + 1)
            patch = img[y0:y1, x0:x1]
            
            if patch.size > 0:
                ph, pw = patch.shape[:2]
                yy, xx = np.mgrid[0:ph, 0:pw]
                dist = np.sqrt((xx - (cx - x0)) ** 2 + (yy - (cy - y0)) ** 2)
                inner_pixels = patch[dist < r_inner]
                
                if len(inner_pixels) > 0:
                    b_mean = int(np.mean(inner_pixels[:, 0]))
                    g_mean = int(np.mean(inner_pixels[:, 1]))
                    r_mean = int(np.mean(inner_pixels[:, 2]))
                    rgb = (r_mean, g_mean, b_mean)
                else:
                    rgb = (0, 0, 0)
            else:
                rgb = (0, 0, 0)
            
            wells_data.append({
                'row': ri,
                'col': ci,
                'center': (int(cx), int(cy)),
                'radius': median_radius,
                'rgb': rgb
            })
            rgb_matrix[ri, ci] = rgb
    
    print(f"✓ Extracted RGB from {len(wells_data)} wells")
    
    return {
        'wells': wells_data,
        'rgb_matrix': rgb_matrix,
        'circles': best_circles,
        'row_centers': sorted(row_centers),
        'col_centers': sorted(col_centers),
        'median_radius': median_radius,
        'image': img
    }, img


# Run analysis
print("="*60)
print("WELL PLATE ANALYSIS - Adaptive Hough Circle Detection")
print("="*60)

camera_folder = Path("camera_test")
jpg_files = sorted(camera_folder.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)

if not jpg_files:
    print("✗ No images found in camera_test folder")
else:
    latest_image = jpg_files[0]
    print(f"\n[1] Analyzing: {latest_image.name}\n")
    
    result, raw_img = detect_wells_adaptive(latest_image)
    
    # ALWAYS show the image, even if detection failed
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    
    if result is not None:
        wells = result['wells']
        rgb_matrix = result['rgb_matrix']
        circles = result['circles']
        row_centers = result['row_centers']
        col_centers = result['col_centers']
        median_radius = result['median_radius']
        
        print("\n[2] Creating visualization...")
        
        # Create annotated image
        img_annotated = img_rgb.copy()
        
        # Draw all detected Hough circles (faint green)
        for (x, y, r) in circles:
            cv2.circle(img_annotated, (int(x), int(y)), int(r), (100, 200, 100), 1)
        
        # Draw clustered well grid (red markers with labels)
        for ri, cy in enumerate(row_centers):
            for ci, cx in enumerate(col_centers):
                cv2.circle(img_annotated, (cx, cy), 5, (255, 0, 0), 2)
                cv2.circle(img_annotated, (cx, cy), 2, (0, 255, 0), -1)
                
                row_letter = chr(65 + ri)
                label = f"{row_letter}{ci + 1}"
                cv2.putText(img_annotated, label, (cx - 8, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0), 1)
        
        # Display side by side
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original Plate Image", fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(img_annotated)
        axes[1].set_title(f"{len(circles)} circles → 8×12 grid (red markers)", fontsize=14)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig("well_detection_result.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print results
        print(f"\n[3] Detection Results:")
        print(f"    Hough circles found: {len(circles)}")
        print(f"    Grid wells: {len(wells)}")
        print(f"    Median radius: {median_radius} px")
        
        print(f"\n[4] Sample RGB values (Row A):")
        for col in range(12):
            rgb = rgb_matrix[0, col]
            print(f"    A{col+1:2d}: RGB={rgb}")
    else:
        # Detection failed - show original image anyway
        print("\n[2] Detection failed - showing original image for inspection...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img_rgb)
        ax.set_title("Original Image (Detection Failed - Manual Annotation May Be Needed)", fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig("well_detection_failed.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\n⚠ MANUAL ANNOTATION RECOMMENDED:")
        print("  The automatic detection could not find 96 wells.")
        print("  Possible causes:")
        print("  - Different plate type or orientation")
        print("  - Camera distance/focus differs from expected")
        print("  - Lighting conditions affect circle edges")

print("\n" + "="*60)
print("✓ ANALYSIS COMPLETE")
print("="*60)


import cv2
import numpy as np
import webcolors
from scipy.spatial import KDTree

# Load your plate image
img_path = "/Users/jeffreyyang/battlebattle/plateImages/capture_1775589733.jpg"  # UPDATE THIS PATH
img = cv2.imread(img_path)

if img is None:
    print(f"Could not load image from {img_path}")
    print("Please update img_path to point to your plate image")
else:
    print(f"Image loaded: {img.shape}")
    
    # Display the image
    from IPython.display import display
    from PIL import Image
    display(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))


# Medium article approach: find radius by testing multiple ranges
def detect_wells_medium_approach(img, target_wells=96):
    """
    Uses the Medium article approach with medianBlur and param1=200, param2=10
    Automatically finds the best radius range.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(gray, 5)
    
    best_circles = None
    best_count = 0
    best_radius = None
    
    # Try different radius ranges (for 96-well plates, typically 10-60px depending on image size)
    for min_r in range(10, 60, 2):
        max_r = min_r + 2  # Tight radius range as per article
        
        circles = cv2.HoughCircles(
            img_blur,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=img.shape[0] / 64,  # Article's approach
            param1=200,
            param2=10,
            minRadius=min_r,
            maxRadius=max_r
        )
        
        if circles is not None:
            count = len(circles[0])
            # We want closest to 96 but not over
            if count >= target_wells:
                print(f"Found {count} circles with radius {min_r}-{max_r}")
                return circles, (min_r, max_r)
            elif count > best_count:
                best_count = count
                best_circles = circles
                best_radius = (min_r, max_r)
    
    print(f"Best found: {best_count} circles with radius {best_radius}")
    return best_circles, best_radius

# Run detection
circles, radius_range = detect_wells_medium_approach(img)

# Visualize results
if circles is not None:
    vis_img = img.copy()
    circles_rounded = np.uint16(np.around(circles))
    
    for i in circles_rounded[0, :96]:  # Limit to first 96
        cv2.circle(vis_img, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Outer circle
        cv2.circle(vis_img, (i[0], i[1]), 2, (0, 0, 255), 3)     # Center point
    
    print(f"Detected {len(circles[0])} circles")
    display(Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)))


# Grid-based well detection: use detected circles to compute grid, then fill all 96 positions

def compute_grid_from_circles(circles, n_cols=12, n_rows=8, merge_threshold=15):
    """
    Use detected circles to compute grid parameters, then generate all 96 well positions.
    merge_threshold: merge coordinates within this distance (prevents duplicates)
    """
    if circles is None or len(circles[0]) < 10:
        print("Need at least 10 detected circles to compute grid")
        return None
    
    # Get circle centers
    centers = [(int(c[0]), int(c[1])) for c in circles[0]]
    
    def merge_nearby(coords, threshold):
        """Merge coordinates that are within threshold of each other"""
        coords = sorted(coords)
        merged = []
        current_group = [coords[0]]
        
        for c in coords[1:]:
            if c - current_group[-1] <= threshold:
                current_group.append(c)
            else:
                merged.append(int(np.mean(current_group)))
                current_group = [c]
        merged.append(int(np.mean(current_group)))
        return merged
    
    # Cluster coordinates into n groups using gaps
    def cluster_coords(coords, n_clusters):
        """Cluster coordinates into n groups using largest gaps"""
        if len(coords) < n_clusters:
            return None
        coords = sorted(coords)
        # Find largest gaps to split into clusters
        gaps = [(coords[i+1] - coords[i], i) for i in range(len(coords)-1)]
        gaps.sort(reverse=True)
        # Take top n_clusters-1 gaps as split points
        split_indices = sorted([g[1] for g in gaps[:n_clusters-1]])
        
        clusters = []
        start = 0
        for idx in split_indices:
            cluster_vals = coords[start:idx+1]
            clusters.append(int(np.mean(cluster_vals)))
            start = idx + 1
        clusters.append(int(np.mean(coords[start:])))
        return sorted(clusters)
    
    # Get all x and y coordinates from detected circles
    all_x = [c[0] for c in centers]
    all_y = [c[1] for c in centers]
    
    # First merge nearby coordinates to remove duplicates
    merged_x = merge_nearby(all_x, merge_threshold)
    merged_y = merge_nearby(all_y, merge_threshold)
    
    print(f"After merging: {len(merged_x)} unique x, {len(merged_y)} unique y positions")
    
    # Cluster into columns and rows
    col_centers = cluster_coords(merged_x, n_cols)
    row_centers = cluster_coords(merged_y, n_rows)
    
    if col_centers is None or row_centers is None:
        # Fallback: compute spacing from detected points
        print("Using spacing-based extrapolation...")
        
        # Compute spacing from merged coordinates
        if len(merged_x) >= 2:
            x_diffs = [merged_x[i+1] - merged_x[i] for i in range(len(merged_x)-1)]
            col_spacing = int(np.median(x_diffs))
        else:
            col_spacing = 45  # Default fallback
            
        if len(merged_y) >= 2:
            y_diffs = [merged_y[i+1] - merged_y[i] for i in range(len(merged_y)-1)]
            row_spacing = int(np.median(y_diffs))
        else:
            row_spacing = 45  # Default fallback
        
        # Find top-left reference point
        min_x = min(all_x)
        min_y = min(all_y)
        
        # Generate grid
        col_centers = [min_x + i * col_spacing for i in range(n_cols)]
        row_centers = [min_y + i * row_spacing for i in range(n_rows)]
    
    print(f"Column positions: {col_centers[:3]}...{col_centers[-1]}")
    print(f"Row positions: {row_centers[:3]}...{row_centers[-1]}")
    print(f"Col spacing: ~{col_centers[1]-col_centers[0]}px, Row spacing: ~{row_centers[1]-row_centers[0]}px")
    
    # Generate all 96 well positions
    well_positions = []
    avg_radius = int(np.mean([c[2] for c in circles[0]]))
    
    for row_idx, y in enumerate(row_centers):
        for col_idx, x in enumerate(col_centers):
            well_positions.append({
                'x': x,
                'y': y,
                'radius': avg_radius,
                'row': chr(ord('A') + row_idx),
                'col': col_idx + 1,
                'name': f"{chr(ord('A') + row_idx)}{col_idx + 1}"
            })
    
    return well_positions, avg_radius

# Compute grid from detected circles (uses 'circles' from cell 3)
well_positions, avg_radius = compute_grid_from_circles(circles)

# Visualize the computed grid
if well_positions:
    vis_grid = img.copy()
    
    for well in well_positions:
        cv2.circle(vis_grid, (well['x'], well['y']), avg_radius, (0, 255, 0), 2)
        cv2.circle(vis_grid, (well['x'], well['y']), 2, (0, 0, 255), 3)
        # Add well name label
        cv2.putText(vis_grid, well['name'], (well['x']-15, well['y']-avg_radius-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
    
    print(f"\nGenerated {len(well_positions)} well positions")
    display(Image.fromarray(cv2.cvtColor(vis_grid, cv2.COLOR_BGR2RGB)))


# Extract RGB values and color names from grid positions (cell 4)

def convert_rgb_to_color_name(rgb_input):
    """Convert RGB values to closest CSS3 color name using KDTree"""
    # Handle different webcolors versions
    try:
        hexnames = webcolors.CSS3_HEX_TO_NAMES
    except AttributeError:
        # Newer webcolors versions use different attribute names
        hexnames = {webcolors.name_to_hex(name): name for name in webcolors.names('css3')}
    
    names = []
    positions = []
    
    for hex_val, name in hexnames.items():
        names.append(name)
        positions.append(webcolors.hex_to_rgb(hex_val))
    
    spacedb = KDTree(positions)
    dist, index = spacedb.query(rgb_input)
    return names[index]

def extract_well_data_from_grid(img, well_positions, sample_radius=5):
    """Extract RGB values and color names for each well using grid positions"""
    labeled_wells = []
    
    for well in well_positions:
        x, y = well['x'], well['y']
        
        # Sample a small region around center for robust color
        y_start = max(0, y - sample_radius)
        y_end = min(img.shape[0], y + sample_radius)
        x_start = max(0, x - sample_radius)
        x_end = min(img.shape[1], x + sample_radius)
        
        region = img[y_start:y_end, x_start:x_end]
        
        # Average BGR, convert to RGB
        avg_bgr = np.mean(region, axis=(0, 1))
        r_val, g, b = int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0])
        
        color_name = convert_rgb_to_color_name((r_val, g, b))
        
        labeled_wells.append({
            'well_name': well['name'],
            'well_coordinates': (x, y),
            'radius': well['radius'],
            'rgb': (r_val, g, b),
            'well_color': color_name
        })
    
    return labeled_wells

# Run extraction using grid positions from cell 4
if well_positions:
    labeled_wells = extract_well_data_from_grid(img, well_positions)
    
    # Print first column (A1-H1)
    print("Well data (first column A1-H1):")
    for well in labeled_wells[:8]:
        print(f"  {well['well_name']}: RGB{well['rgb']} -> {well['well_color']}")
else:
    print("No well positions found. Run cell 4 first.")


# Create 8x12 RGB matrix from labeled wells and visualize

def create_rgb_matrix(labeled_wells):
    """Convert labeled wells to 8x12x3 RGB matrix"""
    rgb_matrix = np.zeros((8, 12, 3), dtype=np.uint8)
    
    for well in labeled_wells:
        # Parse well name (e.g., "A1" -> row 0, col 0)
        row = ord(well['well_name'][0]) - ord('A')
        col = int(well['well_name'][1:]) - 1
        rgb_matrix[row, col] = well['rgb']
    
    return rgb_matrix

# Generate matrix from labeled_wells (cell 5)
if 'labeled_wells' in dir() and labeled_wells:
    rgb_matrix = create_rgb_matrix(labeled_wells)
    print("RGB Matrix shape:", rgb_matrix.shape)
    print("\nRGB values for column 1 (A1-H1):")
    for row in range(8):
        well_name = f"{chr(ord('A')+row)}1"
        print(f"  {well_name}: {tuple(rgb_matrix[row, 0])}")
    
    # Visualize the matrix as a color plate and show the original image side-by-side
    import matplotlib.pyplot as plt
    from PIL import Image
    import cv2
    
    # Load the original image (ensure path is correct)
    orig_img_path = "camera_test/camera_test_20260331135106_hd.jpg"
    orig_img = cv2.imread(orig_img_path)
    if orig_img is not None:
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    else:
        orig_img_rgb = None
    
    # Create a larger visualization for the color matrix
    vis_matrix = np.zeros((8*30, 12*30, 3), dtype=np.uint8)
    for r in range(8):
        for c in range(12):
            vis_matrix[r*30:(r+1)*30, c*30:(c+1)*30] = rgb_matrix[r, c]
    
    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Show original image
    if orig_img_rgb is not None:
        axes[0].imshow(orig_img_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, "Image not found", ha='center', va='center', fontsize=14)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
    
    # Show color matrix
    axes[1].imshow(vis_matrix)
    axes[1].set_title("Extracted Well Colors (8x12 Grid)")
    axes[1].set_xlabel("Column (1-12)")
    axes[1].set_ylabel("Row (A-H)")
    axes[1].set_xticks([15 + c*30 for c in range(12)])
    axes[1].set_xticklabels([str(c+1) for c in range(12)])
    axes[1].set_yticks([15 + r*30 for r in range(8)])
    axes[1].set_yticklabels([chr(ord('A')+r) for r in range(8)])
    plt.tight_layout()
    plt.show()
else:
    print("No labeled_wells found. Run cell 5 first.")


# Export results to CSV/JSON

import json

# Save labeled wells to JSON
if 'labeled_wells' in dir() and labeled_wells:
    output_path = "well_detection_results.json"
    
    # Convert to serializable format
    results = {
        'image_path': img_path,
        'num_wells': len(labeled_wells),
        'wells': labeled_wells
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(labeled_wells)} well results to {output_path}")
    
    # Also print full summary by column
    print("\n=== All Wells Summary ===")
    for col in range(1, 13):
        # Get wells where column number matches exactly
        col_wells = [w for w in labeled_wells if int(w['well_name'][1:]) == col]
        col_wells_sorted = sorted(col_wells, key=lambda w: w['well_name'][0])  # Sort by row letter
        colors = [w['well_color'] for w in col_wells_sorted]
        print(f"Column {col}: {colors}")
else:
    print("No labeled_wells to export. Run cells 4-6 first.")