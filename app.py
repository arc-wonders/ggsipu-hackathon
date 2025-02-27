import os
import cv2
import torch
import numpy as np
import logging
import subprocess
from threading import Thread
from flask import Flask, request, jsonify, Response, abort
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

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" for better accuracy

def repackage_video(input_path, output_path):
    """Use FFmpeg to repackage video and fix metadata issues."""
    command = [
        'ffmpeg',
        '-i', input_path,
        '-c', 'copy',
        '-movflags', 'faststart',
        output_path
    ]
    subprocess.run(command, check=True)

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = model.names[int(box.cls[0])]

                if confidence > 0.4:  # Only consider objects with high confidence
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    logging.info(f"Processed video saved at: {output_path}")

    # Repackage the video with FFmpeg to fix metadata (e.g., duration)
    temp_output = output_path.replace('.mp4', '_temp.mp4')
    repackage_video(output_path, temp_output)
    os.replace(temp_output, output_path)

def process_video_threaded(input_path, output_path):
    thread = Thread(target=process_video, args=(input_path, output_path))
    thread.start()

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

if __name__ == '__main__':
    app.run(debug=True)
