from flask import Flask, jsonify
import cv2
import time
import os

app = Flask(__name__)

# Folder where all images will be saved
SAVE_FOLDER = "REPLACE_WITH_YOUR_FOLDER_PATH"
os.makedirs(SAVE_FOLDER, exist_ok=True)

@app.route('/capture', methods=['POST'])
def capture():
    # Capture image from USB camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Camera capture failed"}), 500

    # Save image as JPG with timestamp in the chosen folder
    timestamp = int(time.time())
    filename = os.path.join(SAVE_FOLDER, f"capture_{timestamp}.jpg")
    cv2.imwrite(filename, frame)

    # Example processing (replace with your logic)
    pipette_volume = 50

    # Return JSON to OT-2
    return jsonify({"pipette_volume": pipette_volume, "image_file": filename})

if __name__ == '__main__':
    app.run(host="REPLACE_WITH_IP", port=5000)