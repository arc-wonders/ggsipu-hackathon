from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
from sort import Sort  

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load YOLO model
model = YOLO("yolov8n.pt")
tracker = Sort()

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    # Process the video
    results = process_video(filename)

    return jsonify({"message": "Processing complete", "results": results})

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    total_speeds = []
    congestion_level = "Green (Low Traffic)"
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = []
        for r in results:
            for box in r.boxes:
                box_coords = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())

                if class_id in [2, 3, 5, 7]:  # Cars, motorcycles, buses, trucks
                    x1, y1, x2, y2 = map(int, box_coords)
                    detections.append([x1, y1, x2, y2, confidence])

        detections = np.array(detections)
        tracked_objects = tracker.update(detections)

        # Estimate speed (Simplified for demo)
        speeds = [np.random.randint(20, 80) for _ in tracked_objects]  # Fake random speed
        total_speeds.extend(speeds)

    cap.release()
    
    avg_speed = np.mean(total_speeds) if total_speeds else 0
    
    # Define congestion levels
    if len(total_speeds) > 50:
        congestion_level = "Red (Heavy Traffic)"
    elif len(total_speeds) > 20:
        congestion_level = "Orange (Moderate Traffic)"

    return {"avg_speed": avg_speed, "congestion_level": congestion_level}

if __name__ == '__main__':
    app.run(debug=True, port=5000)
