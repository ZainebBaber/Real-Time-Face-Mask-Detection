from ultralytics import YOLO
import torch

def main():
    print("Gpu Available:" , torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))

   

    model=YOLO("yolo26n.pt")
    model.train(
        data="face-mask-1/data.yaml",
        epochs=5,
        imgsz=640,
        workers=2
    )



    metrics=model.val()



    results=model("face-mask-1/test/images/14_png_jpg.rf.44e86c7dd6c3e2df74f58ca759af83ae.jpg")
    results[0].show()

    model.export(format="onnx")

if __name__ =="__main__":
    main()