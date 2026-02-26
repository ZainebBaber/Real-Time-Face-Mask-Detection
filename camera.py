import cv2
from ultralytics import YOLO
import time

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Class names for face mask detection (adjust if your classes differ)
CLASS_NAMES = {0: "Mask", 1: "NoMask"}

# Colors for each class (BGR format)
COLORS = {
    0: (0, 255, 0),    # Green for With Mask
    1: (0, 0, 255),    # Red for Without Mask
}

def run_detection():
    # Try different camera indices if 0 doesn't work
    cap = None
    for cam_index in [0, 1, 2]:
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened():
            print(f"Camera opened successfully at index {cam_index}")
            break
    
    if not cap or not cap.isOpened():
        print("ERROR: Could not open any camera.")
        return

    # Force camera settings to avoid black screen
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640 ) #480
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frame

    # # Warm up the camera (first few frames are often black)
    # print("Warming up camera...")
    # for _ in range(5):
    #     cap.read()

    print("Starting detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Failed to grab frame, retrying...")
            continue

        # Check if frame is black (all zeros)
        if frame.sum() == 0:
            print("Black frame received, skipping...")
            continue

        start_time = time.time()


        # Run YOLO inference
        results = model(frame, imgsz=640, conf=0.5, verbose=False)

        end_time = time.time()
        fps = 1 / (end_time - start_time)

        cv2.putText(frame, f"FPS: {int(fps)}", (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Manually draw bounding boxes
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Get label and color
                    label = CLASS_NAMES.get(cls, f"Class {cls}")
                    color = COLORS.get(cls, (255, 255, 255))

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label background
                    label_text = f"{label}: {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)

                    # Draw label text
                    cv2.putText(frame, label_text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show FPS on frame
        cv2.putText(frame, "Press Q to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Display the frame
        cv2.imshow("Face Mask Detection", frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection()
