from dataclasses import dataclass
from pathlib import Path

@dataclass
class AppConfig:
    # Paths
    car_model_path: str = "yolo11n.pt"  # COCO model (car detection)
    plate_model_path: str = "plate.pt"  # license plate model (you can download once)
    seatbelt_model_path: str = "runs/detect/seatbelt_detector7/weights/best.pt"

    # Inference
    device: str = "cpu"     # "cpu" or "0" for GPU
    conf: float = 0.35
    iou: float = 0.5

    # Output
    out_dir: str = "outputs"
    save_crops: bool = True

    # Tesseract (Windows)
    # لو تيسراكت ما هو في PATH حط مساره هنا:
    # مثال: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    tesseract_cmd: str | None = None

    # Which classes to treat as vehicles in COCO
    vehicle_names = {"car", "truck", "bus", "motorcycle"}

    # Seatbelt decision logic (based on your trained model class names)
    safe_keywords = ("seatbelt", "person-seatbelt")
    violation_keywords = ("noseatbelt", "person-noseatbelt")
