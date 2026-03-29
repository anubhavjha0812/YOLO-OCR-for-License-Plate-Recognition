# Automatic Number Plate Recognition (ANPR) – YOLO + EasyOCR

## Overview

This project implements an end-to-end **Automatic Number Plate Recognition (ANPR)** pipeline using:

* YOLO (Ultralytics) → detection
* EasyOCR → text extraction
* OpenCV → video processing

The system processes a video, detects license plates, extracts text, stabilizes predictions, and outputs an annotated video.

---

## Features

* License plate detection using YOLO
* OCR with preprocessing (grayscale + thresholding)
* Character correction (handles OCR noise like 0↔O, 1↔I)
* Temporal stabilization (majority voting across frames)
* Zoomed overlay of detected plates
* Structured logging system
* CPU compatible (no GPU required)

---

## Project Structure

```
.
├── main.py
├── requirements.txt
├── license_plate_best.pt
├── vehicle_video.mp4
├── output_with_licensev_final.mp4
└── README.md
```

---

## Installation

### 1. Create Virtual Environment

```
python3 -m venv venv
source venv/bin/activate
```

---

### 2. Install Dependencies

```
pip install -r requirements.txt
```

---

### 3. Install PyTorch (CPU)

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## requirements.txt

```
opencv-python
numpy
ultralytics
easyocr
```

---

## Usage

```
python main.py
```

Output:

```
output_with_licensev_final.mp4
```

---

## Pipeline Workflow

1. Read video frame-by-frame
2. Detect plates using YOLO
3. Crop plate region
4. Preprocess image for OCR
5. Extract text using EasyOCR
6. Correct OCR errors
7. Validate format using regex
8. Stabilize predictions across frames
9. Overlay results on video

---

## Logging System

The system logs important events:

* INFO → progress updates
* WARNING → non-critical issues
* ERROR → failures

Example:

```
INFO | Starting inference loop...
INFO | Processed 100 frames
WARNING | Invalid bounding box skipped
ERROR | OCR error: ...
```

---

## Configuration

| Parameter    | Description          | Default |
| ------------ | -------------------- | ------- |
| CONF_THRESH  | Detection confidence | 0.3     |
| FPS fallback | Used if FPS invalid  | 25      |
| Buffer size  | Stabilization window | 10      |

---

## GPU Support (Optional Upgrade)

### 1. Install GPU PyTorch

```
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio
```

---

### 2. Enable GPU in EasyOCR

Modify:

```
reader = easyocr.Reader(['en'], gpu=False)
```

Change to:

```
reader = easyocr.Reader(['en'], gpu=True)
```

---

### 3. Move YOLO to GPU

Modify:

```
model = YOLO("license_plate_best.pt")
```

Change to:

```
model = YOLO("license_plate_best.pt")
model.to("cuda")
```

---

### 4. Verify GPU usage

```
python -c "import torch; print(torch.cuda.is_available())"
```

Expected:

```
True
```

---

## Important Notes (GPU)

* GPU must support CUDA
* Proper NVIDIA drivers required
* If CUDA not available → fallback to CPU automatically

---

## Troubleshooting

### cv2 not found

```
pip install opencv-python
```

---

### GPU libraries installing unintentionally

Use CPU index:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

### Output video too short

* Check FPS handling
* Ensure full frame processing (logs should show total frames)

---

### No display (imshow error)

* Install GUI version:

```
pip install opencv-python
```

or disable `cv2.imshow()`

---

## Limitations

* Regex supports only fixed 7-character plates
* No advanced tracking (uses coordinate-based ID)
* OCR accuracy depends on image quality

---

## Future Improvements

* DeepSORT tracking
* Multi-format plate support (India, EU)
* Real-time webcam integration
* API deployment (FastAPI)
* Performance optimization with GPU

---

## Conclusion

This project provides a **complete ANPR pipeline** combining detection, OCR, and stabilization with production-level logging and CPU compatibility.

---
