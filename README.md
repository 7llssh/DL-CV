تمام 👍
هذا **README.md كامل في رسالة وحدة** — انسخه **نسخ-لصق مرة وحدة** واحفظه باسم `README.md` 👇

---

```markdown
# 🚗 Seatbelt Violation Detection System

An end-to-end Computer Vision system for detecting **seatbelt violations** in vehicles.
The project integrates multiple YOLO-based models into a single pipeline that performs:

- Vehicle detection  
- License plate detection  
- License plate OCR  
- Seatbelt detection (Safe / Violation)  

Built using **Python**, **YOLO (Ultralytics)**, **OpenCV**, and **Tesseract OCR**.

---

## 📌 Features

- Detects vehicles in an image
- Crops each detected vehicle individually
- Detects license plates inside vehicles
- Reads plate numbers using OCR
- Detects seatbelt usage
- Classifies driver as:
  - ✅ SAFE (Seatbelt detected)
  - ❌ VIOLATION (No seatbelt detected)
- Outputs annotated images with results

---

## 🗂️ Project Structure

```

DL-CV/
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── detectors.py
│   ├── ocr.py
│   └── pipeline.py
├── main.py
├── requirements.txt
├── README.md
├── dataset/          # Not uploaded to GitHub
├── models/
│   └── plate.pt
├── runs/             # YOLO training outputs
└── outputs/

````

---

## ⚙️ Requirements

- Python **3.10 or 3.11**
- Windows / Linux / macOS
- Tesseract OCR (installed separately)

---

## 📦 Installation

### 1️⃣ Create and activate virtual environment
```bash
python -m venv .venv
````

**PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Git Bash**

```bash
source .venv/Scripts/activate
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Install Tesseract OCR (Windows)

Download from:
[https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

Default path:

```
C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

## ▶️ Running the Project

### Run on a single image

```bash
python main.py \
  --image dataset/test/images/IMAGE_NAME.jpg \
  --plate-model models/plate.pt \
  --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## 📤 Output

* Annotated image saved to:

  ```
  outputs/annotated.jpg
  ```

* Example results:

  ```
  SAFE | Plate: ABC123
  VIOLATION | Plate: XYZ987
  ```

---

## 🧠 Models Used

* **Vehicle Detection**: YOLO (COCO pretrained)
* **License Plate Detection**: Custom YOLO model (`plate.pt`)
* **Seatbelt Detection**: Custom-trained YOLO model
* **OCR**: Tesseract OCR

---

## 🚫 Notes

* `dataset/` and `runs/` folders are excluded from GitHub due to size
* Trained model weights (`.pt`) are optional to upload
* Update paths in `config.py` if needed

---

## 📊 Future Improvements

* Real-time video / webcam support
* Multi-camera traffic violation system
* Improved OCR accuracy
* Integration with traffic databases

---


