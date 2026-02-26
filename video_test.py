# video_inference_resized.py
from ultralytics import YOLO
import cv2
import os

def annotate_video(model_path, video_path, output_path="annotated_video.mp4", target_size=640):
    """
    Annotate a video using a trained YOLO model with bounding boxes and labels.
    
    Args:
        model_path (str): Path to your trained YOLO model (best.pt)
        video_path (str): Path to the input video
        output_path (str): Path to save the annotated video
        target_size (int): Resize frames to this size (square), e.g., 640
    """
    # 1️⃣ Load trained YOLO model
    model = YOLO(model_path)

    # 2️⃣ Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # 3️⃣ Get original fps
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 4️⃣ Prepare video writer with resized frame size
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (target_size, target_size)  # width x height
    )

    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # 5️⃣ Resize frame to target_size x target_size
        frame_resized = cv2.resize(frame, (target_size, target_size))

        # 6️⃣ Run YOLO inference
        results = model(frame_resized)

        # 7️⃣ Annotate frame with bounding boxes and labels
        annotated_frame = results[0].plot()

        # 8️⃣ Write annotated frame to output video
        out.write(annotated_frame)

    # 9️⃣ Release everything
    cap.release()
    out.release()
    print(f"Annotated video saved at: {output_path}")


if __name__ == "__main__":
    model_path = "runs/detect/train/weights/best.pt"
    video_path = "7585010-uhd_2160_4096_25fps.mp4"
    output_path = "annotated_video.mp4"

    annotate_video(model_path, video_path, output_path, target_size=640)