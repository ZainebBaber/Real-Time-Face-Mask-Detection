import os
from pathlib import Path
import cv2
import time
from ultralytics import YOLO


BASE_DIR=Path(__file__).resolve().parent.parent
MODEL_PATH=BASE_DIR /"runs/detect/train/weights/best.onnx"

CLASS_NAMES= {0: "Mask", 1: "NoMask"}
COLORS={
    0: (0,255,0), #green
    1: (0,0,255)
}

model=YOLO(MODEL_PATH)

def get_camera():
    for cam_index in [0,1,2]:
        cap=cv2.VideoCapture(cam_index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

    return None


def process_frame(frame):
    start_time=time.time()

    results=model(frame, imgsz=640, conf=0.5, verbose=False)
    fps=1/(time.time()-start_time)

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                label = CLASS_NAMES.get(cls, f"Class {cls}")
                color = COLORS.get(cls, (255, 255, 255))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label_text = f"{label}: {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    
    return frame
