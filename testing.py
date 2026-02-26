from ultralytics import YOLO


model=YOLO("best.pt")

metrics=model.val(data="Face-mask-detection-1/data.yaml", split="test")
