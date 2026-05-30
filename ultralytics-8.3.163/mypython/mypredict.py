from ultralytics import YOLO

model = YOLO(r"D:\deeplearning\ultralytics-8.3.163\results\yolov8n\weights\best.pt")
model.predict(                                                                                     
source=r"D:\google下载\摔倒参考 _ 50种摔倒方式.mp4",
save=False,
show=True,
)