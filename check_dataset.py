from ultralytics import YOLO

# تحميل مودل YOLO خفيف كبداية
model = YOLO("yolo11n.pt")

# تدريب المودل
model.train(
    data="dataset/modified_data.yaml",
    epochs=3,
    imgsz=640,
    batch=16,
    name="seatbelt_detector"
)
