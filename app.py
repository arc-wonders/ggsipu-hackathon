import os
import json
import time
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed_videos'
ANALYTICS_FILE = 'analytics.json'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB Limit

model = YOLO("yolov8n.pt")

road_width_m = 32  
road_length_m = 140  
pts_src = np.array([[550, 350], [750, 350], [1200, 700], [200, 700]], dtype=np.float32)
pts_dst = np.array([[0, road_length_m], [road_width_m, road_length_m], [road_width_m, 0], [0, 0]], dtype=np.float32)

homography_matrix, _ = cv2.findHomography(pts_src, pts_dst)

class CustomSORT:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        updated_tracks = {}
        for det in detections:
            x1, y1, x2, y2, conf = det
            found = False
            for track_id, (tx1, ty1, tx2, ty2) in self.tracks.items():
                iou = self.calculate_iou((x1, y1, x2, y2), (tx1, ty1, tx2, ty2))
                if iou > 0.3:
                    updated_tracks[track_id] = (x1, y1, x2, y2)
                    found = True
                    break
            if not found:
                updated_tracks[self.next_id] = (x1, y1, x2, y2)
                self.next_id += 1
        self.tracks = updated_tracks
        return [(x1, y1, x2, y2, track_id) for track_id, (x1, y1, x2, y2) in self.tracks.items()]
    
    @staticmethod
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x1_t, y1_t, x2_t, y2_t = box2
        inter_x1 = max(x1, x1_t)
        inter_y1 = max(y1, y1_t)
        inter_x2 = min(x2, x2_t)
        inter_y2 = min(y2, y2_t)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_t - x1_t) * (y2_t - y1_t)
        return inter_area / (box1_area + box2_area - inter_area)

tracker = CustomSORT()
object_speeds = {}
object_speed_history = {}

def smooth_speed(track_id, new_speed, history_length=5):
    if track_id not in object_speed_history:
        object_speed_history[track_id] = []
    object_speed_history[track_id].append(new_speed)
    if len(object_speed_history[track_id]) > history_length:
        object_speed_history[track_id].pop(0)
    return sum(object_speed_history[track_id]) / len(object_speed_history[track_id])

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
        process_video(filepath, processed_path)
        return jsonify({'message': 'Processing Complete', 'processed_video': f'/processed_videos/{file.filename}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        detections = []
        vehicle_count = 0
        
        for r in results:
            for box in r.boxes:
                box_coords = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                if class_id in [2, 3, 5, 7]:
                    x1, y1, x2, y2 = map(int, box_coords)
                    detections.append([x1, y1, x2, y2, confidence])
                    vehicle_count += 1
        
        tracked_objects = tracker.update(detections)
        
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            pixel_center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
            real_world_center = cv2.perspectiveTransform(pixel_center.reshape(1, 1, 2), homography_matrix).reshape(2)
            real_x, real_y = real_world_center

            if track_id in object_speeds:
                prev_x, prev_y, prev_time = object_speeds[track_id]
                time_diff = time.time() - prev_time
                if time_diff > 0:
                    distance_moved = np.linalg.norm([real_x - prev_x, real_y - prev_y])
                    speed_kmh = (distance_moved / time_diff) * 3.6  
                    stable_speed = smooth_speed(track_id, speed_kmh)
                    object_speeds[track_id] = (real_x, real_y, time.time())
        
        out.write(frame)
    
    cap.release()
    out.release()

@app.route('/processed_videos/<filename>')
def get_processed_video(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
