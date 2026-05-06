from ultralytics import YOLO
model = YOLO("models/yolo_best_yolo26s.pt")
metrics = model.val(data="data/yolo/clearsar.yaml", split="val", augment=True, imgsz=768)
print(metrics.box.map)