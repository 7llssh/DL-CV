from __future__ import annotations
import os
from pathlib import Path
import cv2
import numpy as np

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_image_bgr(image_path: str | Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img

def save_image(path: str | Path, img_bgr: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img_bgr)

def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2

def crop(img_bgr: np.ndarray, box_xyxy) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
    return img_bgr[y1:y2, x1:x2].copy()

def draw_box(img_bgr: np.ndarray, box_xyxy, label: str, thickness: int = 2):
    x1, y1, x2, y2 = map(int, box_xyxy)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    if label:
        cv2.putText(img_bgr, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
