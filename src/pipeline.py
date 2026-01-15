from __future__ import annotations
from pathlib import Path
import cv2

from .config import AppConfig
from .utils import ensure_dir, read_image_bgr, save_image, crop, draw_box
from .detectors import load_model, get_name_map
from .ocr import setup_tesseract, ocr_plate_bgr

class SeatbeltPipeline:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        setup_tesseract(cfg.tesseract_cmd)

        self.car_model = load_model(cfg.car_model_path, cfg.device)
        self.plate_model = load_model(cfg.plate_model_path, cfg.device)
        self.seatbelt_model = load_model(cfg.seatbelt_model_path, cfg.device)

        self.car_names = get_name_map(self.car_model)
        self.plate_names = get_name_map(self.plate_model)
        self.seatbelt_names = get_name_map(self.seatbelt_model)

        self.out_dir = ensure_dir(cfg.out_dir)
        self.crops_dir = ensure_dir(self.out_dir / "crops")
        self.plates_dir = ensure_dir(self.out_dir / "plates")

    def _is_vehicle(self, cls_id: int) -> bool:
        name = self.car_names.get(int(cls_id), "")
        return name in self.cfg.vehicle_names

    def _seatbelt_status(self, detections) -> str:
        """
        Return: "SAFE" or "VIOLATION" or "UNKNOWN"
        Based on keywords in your seatbelt model class names.
        """
        found_safe = False
        found_violation = False

        for d in detections:
            cls_id = int(d["cls"])
            name = self.seatbelt_names.get(cls_id, "").lower()
            if any(k in name for k in self.cfg.violation_keywords):
                found_violation = True
            if any(k in name for k in self.cfg.safe_keywords):
                found_safe = True

        if found_violation:
            return "VIOLATION"
        if found_safe:
            return "SAFE"
        return "UNKNOWN"

    def run_on_image(self, image_path: str) -> dict:
        img = read_image_bgr(image_path)
        original = img.copy()
        H, W = img.shape[:2]

        # 1) Detect vehicles in the full image
        car_res = self.car_model.predict(
            source=img,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            device=self.cfg.device,
            verbose=False
        )[0]

        vehicles = []
        for b in car_res.boxes:
            cls_id = int(b.cls.item())
            if not self._is_vehicle(cls_id):
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            vehicles.append({"box": (x1, y1, x2, y2), "cls": cls_id, "conf": float(b.conf.item())})

        results = []
        for i, v in enumerate(vehicles, 1):
            car_crop = crop(img, v["box"])
            car_crop_path = self.crops_dir / f"car_{i}.jpg"
            if self.cfg.save_crops:
                save_image(car_crop_path, car_crop)

            # 2) Detect plate inside car crop
            plate_text = ""
            plate_box = None

            plate_res = self.plate_model.predict(
                source=car_crop,
                conf=max(0.25, self.cfg.conf),
                iou=self.cfg.iou,
                device=self.cfg.device,
                verbose=False
            )[0]

            # pick best plate (highest conf)
            best_plate = None
            if plate_res.boxes is not None and len(plate_res.boxes) > 0:
                best_plate = max(plate_res.boxes, key=lambda bb: float(bb.conf.item()))

            if best_plate is not None:
                px1, py1, px2, py2 = best_plate.xyxy[0].tolist()
                plate_box = (px1, py1, px2, py2)
                plate_crop = crop(car_crop, plate_box)
                plate_path = self.plates_dir / f"plate_{i}.jpg"
                if self.cfg.save_crops:
                    save_image(plate_path, plate_crop)
                plate_text = ocr_plate_bgr(plate_crop)

            # 3) Seatbelt detection inside car crop
            seat_res = self.seatbelt_model.predict(
                source=car_crop,
                conf=self.cfg.conf,
                iou=self.cfg.iou,
                device=self.cfg.device,
                verbose=False
            )[0]

            dets = []
            if seat_res.boxes is not None:
                for bb in seat_res.boxes:
                    sx1, sy1, sx2, sy2 = bb.xyxy[0].tolist()
                    dets.append({
                        "box": (sx1, sy1, sx2, sy2),
                        "cls": int(bb.cls.item()),
                        "conf": float(bb.conf.item())
                    })

            status = self._seatbelt_status(dets)

            # 4) Draw results on original image
            # Vehicle box on original
            draw_box(original, v["box"], f"Vehicle {i}")

            # Add text near vehicle box
            x1, y1, x2, y2 = map(int, v["box"])
            label = f"{status}"
            if plate_text:
                label += f" | Plate: {plate_text}"
            cv2.putText(original, label, (x1, min(H - 10, y2 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if status=="VIOLATION" else (0, 255, 0),
                        2, cv2.LINE_AA)

            results.append({
                "vehicle_index": i,
                "vehicle_box": v["box"],
                "plate_text": plate_text,
                "status": status,
                "car_crop": str(car_crop_path),
            })

        out_path = self.out_dir / "annotated.jpg"
        save_image(out_path, original)

        return {
            "image": image_path,
            "annotated": str(out_path),
            "vehicles_found": len(vehicles),
            "results": results,
        }
