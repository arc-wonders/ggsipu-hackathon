import os
import json
import time
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from sort import Sort  

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed_videos'
ANALYTICS_FILE = 'analytics.json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load YOLO Model
model = YOLO("yolov8n.pt")
tracker = Sort()

# Load existing analytics data
if os.path.exists(ANALYTICS_FILE):
    with open(ANALYTICS_FILE, 'r') as f:
        try:
            analytics_data = json.load(f)
        except json.JSONDecodeError:
            analytics_data = []
else:
    analytics_data = []

def save_analytics():
    """Save analytics data to JSON file."""
    with open(ANALYTICS_FILE, 'w') as f:
        json.dump(analytics_data, f, indent=4)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    processed_path = os.path.join(PROCESSED_FOLDER, file.filename)
    
    try:
        congestion_data, speed_data = process_video(filepath, processed_path)

        # Store analytics as a list, not a dictionary
        analytics_data.append({
            "video": file.filename,
            "timestamp": time.time(),
            "congestion_vs_time": congestion_data,
            "speed_vs_time": speed_data
        })
        save_analytics()
        
        return jsonify({'message': 'Processing Complete', 'processed_video': f'/processed_videos/{file.filename}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    congestion_data = []
    speed_data = []
    object_speeds = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        detections = []
        num_vehicles = 0
        total_speed = 0
        
        for r in results:
            for box in r.boxes:
                box_coords = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())

                if class_id in [2, 3, 5, 7]:  # Vehicles only
                    x1, y1, x2, y2 = map(int, box_coords)
                    detections.append([x1, y1, x2, y2, confidence])
        
        detections = np.array(detections)
        tracked_objects = tracker.update(detections)
        
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj.astype(int)
            num_vehicles += 1
            speed = object_speeds.get(track_id, 30)  # Placeholder speed
            total_speed += speed
        
        avg_speed = total_speed / num_vehicles if num_vehicles > 0 else 0
        congestion_data.append({'time': time.time(), 'vehicles': num_vehicles})
        speed_data.append({'time': time.time(), 'speed': avg_speed})
        
        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return congestion_data, speed_data

@app.route('/list_videos')
def list_videos():
    return jsonify([entry["video"] for entry in analytics_data]), 200

@app.route('/analytics/<video_filename>')
def get_analytics(video_filename):
    for entry in analytics_data:
        if entry["video"] == video_filename:
            return jsonify(entry), 200
    return jsonify({'error': 'Video not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
