from __future__ import annotations
import re
import cv2
import pytesseract
import numpy as np

def setup_tesseract(tesseract_cmd: str | None):
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

def clean_plate_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^A-Za-z0-9\-]+", "", text)
    return text

def ocr_plate_bgr(plate_bgr: np.ndarray) -> str:
    # تحسين بسيط للوحة قبل OCR
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 5)

    config = "--oem 3 --psm 7"
    text = pytesseract.image_to_string(thr, config=config)
    return clean_plate_text(text)
