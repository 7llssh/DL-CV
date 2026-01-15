import argparse
from src.config import AppConfig
from src.pipeline import SeatbeltPipeline

def parse_args():
    p = argparse.ArgumentParser(description="Car + Plate + OCR + Seatbelt pipeline")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--out", default="outputs", help="Output directory")
    p.add_argument("--device", default="cpu", help='cpu or "0" for GPU')
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    p.add_argument("--car-model", default="yolo11n.pt")
    p.add_argument("--plate-model", default="plate.pt")
    p.add_argument("--seatbelt-model", default="runs/detect/seatbelt_detector7/weights/best.pt")
    p.add_argument("--tesseract-cmd", default=None, help="Full path to tesseract.exe on Windows if needed")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = AppConfig(
        car_model_path=args.car_model,
        plate_model_path=args.plate_model,
        seatbelt_model_path=args.seatbelt_model,
        device=args.device,
        conf=args.conf,
        out_dir=args.out,
        tesseract_cmd=args.tesseract_cmd
    )

    pipe = SeatbeltPipeline(cfg)
    report = pipe.run_on_image(args.image)

    print("\n✅ DONE")
    print("Annotated image:", report["annotated"])
    print("Vehicles found:", report["vehicles_found"])
    for r in report["results"]:
        print(f"- Car {r['vehicle_index']}: {r['status']} | Plate: {r['plate_text']}")

if __name__ == "__main__":
    main()
