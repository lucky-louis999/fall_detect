from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"yolov8n.pt")
    model.train(
        data=r"D:\deeplearning\ultralytics-8.3.163\ultralytics\cfg\datasets\fall_detect.yaml",
        epochs=150,
        imgsz=640,
        batch=-1,
        cache="ram",
        workers=1,
        project="results",
        name="yolov8n"
    )