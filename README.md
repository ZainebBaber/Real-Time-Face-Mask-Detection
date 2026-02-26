# Face Mask Detection using YOLO and FastAPI

A real-time face mask detection project built with **Ultralytics YOLO** and **FastAPI**. Detects faces with and without masks on live video feeds, and displays bounding boxes and labels in a web browser.

---

## Features

- Train YOLO models on custom face mask datasets
- Real-time webcam detection with bounding boxes
- Export models to **ONNX** or **PyTorch** for efficient inference
- FastAPI + WebSocket backend for streaming video to frontend
- Lightweight repo (ignores large dataset images)

---

## Project Structure
├── api/ # FastAPI backend code
├── face-mask-1/ # Dataset folder (only data.yaml tracked)
├── runs/ # YOLO training outputs (models, logs)
├── templates/ # HTML frontend templates
├── training.py # YOLO training script
├── testing.py # Model testing script
├── video_test.py # Script for testing video files
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## Installation

1. Clone the repository:

2. Create and activate a Python environment:

3. Install dependencies:
pip install -r requirements.txt

## Notes

Dataset images are not tracked in the repo to keep it lightweight

data.yaml is included to configure the classes and paths

Model files (best.pt, best.onnx) and runs/ folder are tracked
