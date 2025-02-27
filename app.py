import os
import cv2
import numpy as np
import time
import json
import logging
import subprocess
from threading import Thread
from flask import Flask, request, jsonify, Response, abort, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed_videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB limit

logging.basicConfig(level=logging.INFO)

# -------------------------------
# Global YOLO model
# -------------------------------
model = YOLO("yolov8n.pt")  # or "yolov8s.pt" for better accuracy

# -------------------------------
# Global congestion data storage
# -------------------------------
analysis_data = []  # Will store data for the *most recently processed video*

def reencode_video(input_path, output_path):
    """
    Re-encodes the input video into H.264 (video) + AAC (audio),
    and uses '-movflags faststart' so it can start playing immediately.
    """
    command = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-movflags', 'faststart',
        '-crf', '23',
        output_path
    ]
    print("Running FFmpeg command:", " ".join(command))
    subprocess.run(command, check=True)
    print("Re-encode complete:", output_path)

# -------------------------------
# Custom SORT Tracker
# -------------------------------
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
                iou_val = self.calculate_iou((x1, y1, x2, y2), (tx1, ty1, tx2, ty2))
                if iou_val > 0.3:
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
        union_area = box1_area + box2_area - inter_area + 1e-6
        return inter_area / union_area

# -------------------------------
# Speed Estimation Helpers
# -------------------------------
object_speeds = {}
object_speed_history = {}

def smooth_speed(track_id, new_speed, history_length=5):
    if track_id not in object_speed_history:
        object_speed_history[track_id] = []
    object_speed_history[track_id].append(new_speed)
    if len(object_speed_history[track_id]) > history_length:
        object_speed_history[track_id].pop(0)
    return sum(object_speed_history[track_id]) / len(object_speed_history[track_id])

# -------------------------------
# Main Processing Function
# -------------------------------
def process_video(input_path, output_path):
    global analysis_data
    analysis_data = []  # Reset each time we process a new video

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Homography setup
    road_width_m = 32
    road_length_m = 140
    pts_src = np.array([[550, 350], [750, 350], [1200, 700], [200, 700]], dtype=np.float32)
    pts_dst = np.array([
        [0,   road_length_m],
        [road_width_m, road_length_m],
        [road_width_m, 0],
        [0, 0]
    ], dtype=np.float32)
    homography_matrix, _ = cv2.findHomography(pts_src, pts_dst)

    # SORT tracker
    tracker = CustomSORT()

    global object_speeds, object_speed_history
    object_speeds = {}
    object_speed_history = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start_time = time.time()

        # YOLO detection
        results = model.predict(frame, conf=0.4)
        detections = []
        vehicle_count = 0

        for r in results:
            for box in r.boxes:
                box_coords = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())

                # Track typical vehicles
                if class_id in [2, 3, 5, 7] and confidence > 0.4:
                    x1, y1, x2, y2 = map(int, box_coords)
                    detections.append([x1, y1, x2, y2, confidence])
                    vehicle_count += 1

        # Update tracker
        tracked_objects = tracker.update(detections)
        current_time = time.time()

        # Speed calculations
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            pixel_center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
            real_world_center = cv2.perspectiveTransform(pixel_center.reshape(1, 1, 2), homography_matrix).reshape(2)
            real_x, real_y = real_world_center

            if track_id in object_speeds:
                prev_x, prev_y, prev_time = object_speeds[track_id]
                time_diff = current_time - prev_time
                if time_diff > 0:
                    distance_moved = np.linalg.norm([real_x - prev_x, real_y - prev_y])
                    speed_kmh = (distance_moved / time_diff) * 3.6
                    stable_speed = smooth_speed(track_id, speed_kmh)
                    object_speeds[track_id] = (real_x, real_y, current_time)

                    # Draw bounding box & speed
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{stable_speed:.2f} km/h", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                object_speeds[track_id] = (real_x, real_y, current_time)

        # Congestion overlay
        if vehicle_count > 15:
            crowd_color = (0, 0, 255)
            crowd_text = "HIGH"
        elif vehicle_count > 7:
            crowd_color = (0, 165, 255)
            crowd_text = "MEDIUM"
        else:
            crowd_color = (0, 255, 0)
            crowd_text = "LOW"

        cv2.putText(frame, f"Congestion: {crowd_text}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, crowd_color, 2)

        # FPS overlay
        end_time = time.time()
        fps_display = 1 / (end_time - start_time + 1e-6)
        cv2.putText(frame, f"FPS: {fps_display:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        out.write(frame)

        # Collect data for analysis every second
        if fps > 0 and frame_count % int(fps) == 0:
            current_time_sec = frame_count / fps
            analysis_data.append({
                'time_sec': current_time_sec,
                'vehicle_count': vehicle_count
            })

    cap.release()
    out.release()
    logging.info(f"Processed video saved at: {output_path}")

    # Re-encode the final file
    temp_output = output_path.replace('.mp4', '_temp.mp4')
    reencode_video(output_path, temp_output)
    os.replace(temp_output, output_path)
    logging.info(f"Re-encoded (final) video saved at: {output_path}")

def process_video_threaded(input_path, output_path):
    def worker():
        try:
            process_video(input_path, output_path)
        except subprocess.CalledProcessError as e:
            print("FFmpeg error:", e)
        except Exception as e:
            print("Unexpected error in background thread:", e)

    thread = Thread(target=worker)
    thread.start()

@app.route('/')
def serve_index():
    """
    Serve 'index.html' at the root URL.
    Make sure you have a 'static' folder with 'index.html' inside it.
    """
    return send_from_directory('static', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = file.filename
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(PROCESSED_FOLDER, filename)

    file.save(input_path)
    process_video_threaded(input_path, output_path)

    return jsonify({'message': 'Video uploaded and processing started', 'filename': filename})

@app.route('/processed_videos/<path:filename>')
def get_processed_video(filename):
    processed_folder = os.path.abspath(PROCESSED_FOLDER)
    file_path = os.path.join(processed_folder, filename)

    if not os.path.exists(file_path):
        return abort(404)

    def stream_video():
        with open(file_path, "rb") as video_file:
            while chunk := video_file.read(4096):
                yield chunk

    response = Response(stream_video(), content_type="video/mp4")
    response.headers["Accept-Ranges"] = "bytes"
    return response

@app.route('/list_videos')
def list_videos():
    try:
        videos = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
        return jsonify(videos)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analysis_data', methods=['GET'])
def analysis_data_route():
    """
    Returns the congestion-over-time data (time_sec vs vehicle_count)
    in JSON format for the most recently processed video.
    """
    global analysis_data
    return jsonify(analysis_data)

if __name__ == '__main__':
    app.run(debug=True)
