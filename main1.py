import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
from collections import defaultdict, deque
import os

# --- VALIDATION ---
assert os.path.exists("license_plate_best.pt"), "Model file missing"
assert os.path.exists("vehicle_video.mp4"), "Input video missing"

model = YOLO("license_plate_best.pt")
reader = easyocr.Reader(['en'], gpu=False)

# regex: 2Letters+2NUM+3Letters
plate_pattern = re.compile(r"[A-Z]{2}[0-9]{2}[A-Z]{3}$")

def correct_plate_format(ocr_text):
    mapping_num_to_alpha = {"0": "O", "1": "I", "5": "S", "8": "B"}
    mapping_alpha_to_num = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8"}

    ocr_text = ocr_text.upper().replace(" ", "")
    if len(ocr_text) != 7:
        return ""

    corrected = []
    for i, ch in enumerate(ocr_text):
        if i < 2 or i >= 4:
            if ch.isdigit() and ch in mapping_num_to_alpha:
                corrected.append(mapping_num_to_alpha[ch])
            elif ch.isalpha():
                corrected.append(ch)
            else:
                return ""
        else:
            if ch.isalpha() and ch in mapping_alpha_to_num:
                corrected.append(mapping_alpha_to_num[ch])
            elif ch.isdigit():
                corrected.append(ch)
            else:
                return ""

    return "".join(corrected)

# --- OCR PROCESSING ---
def recognize_plate(plate_crop):
    if plate_crop.size == 0:
        return ""

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plate_resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    try:
        ocr_result = reader.readtext(
            plate_resized, detail=0,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        if len(ocr_result) > 0:
            candidate = correct_plate_format(ocr_result[0])
            if candidate and plate_pattern.match(candidate):
                return candidate
    except Exception as e:
        print("OCR error:", e)

    return ""

# --- STABILIZATION ---
plate_history = defaultdict(lambda: deque(maxlen=10))
plate_final = {}

def get_box_id(x1, y1, x2, y2):
    return f"{int(x1/10)}_{int(y1/10)}_{int(x2/10)}_{int(y2/10)}"

def get_stable_plate(box_id, new_text):
    if new_text:
        plate_history[box_id].append(new_text)
        most_common = max(set(plate_history[box_id]), key=plate_history[box_id].count)
        plate_final[box_id] = most_common
    return plate_final.get(box_id, "")

# --- VIDEO SETUP ---
input_video = "vehicle_video.mp4"
output_video = "output_with_licensev4.mp4"

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (int(cap.get(3)), int(cap.get(4)))
)

CONF_THRESH = 0.3

# --- MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf.item())
            if conf < CONF_THRESH:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Safe bounding
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            plate_crop = frame[y1:y2, x1:x2]

            # OCR
            text = recognize_plate(plate_crop)

            # Stabilization
            box_id = get_box_id(x1, y1, x2, y2)
            stable_text = get_stable_plate(box_id, text)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Overlay zoomed plate
            if plate_crop.size > 0:
                overlay_h, overlay_w = 150, 400
                plate_resized = cv2.resize(plate_crop, (overlay_w, overlay_h))

                oy1 = max(0, y1 - overlay_h - 40)
                ox1 = x1
                oy2, ox2 = oy1 + overlay_h, ox1 + overlay_w

                if oy1 >= 0 and ox1 >= 0 and oy2 <= frame.shape[0] and ox2 <= frame.shape[1]:
                    frame[oy1:oy2, ox1:ox2] = plate_resized

                    if stable_text:
                        cv2.putText(frame, stable_text, (x1, y1 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)
                        cv2.putText(frame, stable_text, (x1, y1 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    out.write(frame)
    cv2.imshow("Annotated Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Annotated video saved as", output_video)