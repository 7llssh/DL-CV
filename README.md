# 🚗 Seatbelt Violation Detection System

An end-to-end Computer Vision system designed to detect seatbelt violations and identify vehicle license plates from images.

The system combines multiple computer vision models into a unified detection pipeline capable of:

- Vehicle Detection
- License Plate Detection
- License Plate Recognition (OCR)
- Seatbelt Usage Detection
- Driver Safety Classification

The project is built using Python, YOLO, OpenCV, and Tesseract OCR.

---

## 📌 Overview

The system processes a traffic image through multiple stages.

1. Detect vehicles in the input image.
2. Crop each detected vehicle.
3. Detect the license plate within each vehicle.
4. Extract the plate number using OCR.
5. Detect whether the driver is wearing a seatbelt.
6. Classify the result as `SAFE` or `VIOLATION`.
7. Generate an annotated output image containing the detection results.

---

## ✨ Features

- 🚘 Vehicle detection using a pretrained YOLO model
- 🔍 Automatic vehicle cropping and processing
- 🔢 License plate detection using a custom YOLO model
- 📝 License plate recognition using Tesseract OCR
- 🦺 Seatbelt usage detection
- ⚠️ Automatic violation classification
- 🖼️ Annotated output image generation
- 🧩 Modular pipeline architecture

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLO / Ultralytics | Object detection |
| OpenCV | Image processing and cropping |
| Tesseract OCR | License plate text recognition |
| NumPy | Image and numerical operations |

---

## 🗂️ Project Structure

```text
DL-CV/
│
├── src/
│   ├── config.py          # Configuration and model paths
│   ├── utils.py           # Helper functions
│   ├── detectors.py       # Detection model logic
│   ├── ocr.py             # License plate OCR processing
│   └── pipeline.py        # Main detection pipeline
│
├── main.py                # Application entry point
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
│
├── dataset/               # Training and testing datasets
├── models/                # Trained model weights
│   └── plate.pt
│
├── runs/                  # YOLO training outputs
└── outputs/               # Annotated detection results
```

> Note: The `dataset/` and `runs/` directories may be excluded from the repository because of their large size.

---

## ⚙️ Requirements

Before running the project, make sure you have:

- Python 3.10 or 3.11
- pip
- Tesseract OCR
- Windows, Linux, or macOS

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd DL-CV
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

#### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Git Bash

```bash
source .venv/Scripts/activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔤 Tesseract OCR Setup

Tesseract OCR must be installed separately.

After installation, verify the executable path.

Example Windows installation path:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

Make sure the correct path is provided when running the project.

---

## ▶️ Running the Project

Run the system on a single image:

```bash
python main.py \
  --image dataset/test/images/IMAGE_NAME.jpg \
  --plate-model models/plate.pt \
  --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

The pipeline will process the image, detect vehicles, analyze seatbelt usage, detect license plates, perform OCR, and save the final annotated result.

---

## 🔄 Detection Pipeline

```text
Input Image
     │
     ▼
Vehicle Detection
     │
     ▼
Vehicle Cropping
     │
     ├───────────────┐
     ▼               ▼
Seatbelt         License Plate
Detection          Detection
     │               │
     │               ▼
     │              OCR
     │               │
     └───────┬───────┘
             ▼
     Result Classification
             │
             ▼
       Annotated Output
```

---

## 📤 Output

The processed image is saved in:

```text
outputs/annotated.jpg
```

Example detection results:

```text
SAFE | Plate: ABC123
VIOLATION | Plate: XYZ987
```

The output image includes:

- Vehicle bounding boxes
- Seatbelt classification
- License plate number
- Violation status

---

## 🧠 Models

### Vehicle Detection

Uses a pretrained YOLO model trained on the COCO dataset to detect vehicles.

### License Plate Detection

Uses a custom-trained YOLO model to locate license plates within detected vehicles.

```text
models/plate.pt
```

### Seatbelt Detection

Uses a custom-trained YOLO model to classify seatbelt usage.

The final classification is:

```text
SAFE
```

or:

```text
VIOLATION
```

### License Plate Recognition

Tesseract OCR is used to extract characters from detected license plate regions.

---

## 🚫 Repository Notes

To keep the repository lightweight:

- Large datasets are not included.
- YOLO training outputs may be excluded.
- Large model weights may be excluded.
- Local environment files should not be committed.

Recommended `.gitignore` entries:

```gitignore
.venv/
__pycache__/
*.pyc

dataset/
runs/
outputs/

.DS_Store
```

---

## 🚀 Future Improvements

Future development may include:

- Real-time video processing
- Live webcam support
- Traffic surveillance camera integration
- Vehicle tracking across video frames
- Improved license plate OCR accuracy
- Support for Arabic license plate recognition
- Automatic violation report generation
- Database integration
- REST API integration
- Web-based monitoring dashboard
- Multi-camera traffic monitoring

---

## 🎯 Use Cases

The system can be adapted for:

- Smart traffic monitoring
- Road safety analysis
- Automated violation detection
- Intelligent transportation systems
- Traffic surveillance systems
- AI-based road safety research

---

## 📄 License

This project is intended for educational and research purposes.
