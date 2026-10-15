# Aman Mourya's Algorithm
# License Plate Recognition using YOLOv8 and OpenCV
from ultralytics import YOLO
import cv2
import os
import numpy as np
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture("car.mp4")

# COCO classes:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

vehicle_classes = [2, 3, 5, 7]
seen_vehicle_ids = set()
saved_vehicle_dir = "saved_vehicles"
os.makedirs(saved_vehicle_dir, exist_ok=True)
frame_number = 0
ocr_cache = {}
OCR_INTERVAL = 12


def process_vehicle(vehicle_crop):
    """Find likely plate regions and OCR several enhanced versions of them."""
    if vehicle_crop is None or vehicle_crop.size == 0:
        return ""

    try:
        # OCR does not need the full-resolution vehicle image.
        max_width = 640
        if vehicle_crop.shape[1] > max_width:
            scale = max_width / vehicle_crop.shape[1]
            vehicle_crop = cv2.resize(
                vehicle_crop, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA
            )
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        image_height, image_width = gray.shape
        regions = [gray]

        # A plate is normally a wide, bright, rectangular region near the
        # lower half of a vehicle. Add contour candidates before OCR.
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(blurred, 60, 180)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Keep only a few plausible candidates; OCR is the expensive operation.
        candidate_regions = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            ratio = width / float(height) if height else 0
            area = width * height
            if (2.0 <= ratio <= 6.5 and
                    0.01 * image_width * image_height <= area <=
                    0.5 * image_width * image_height and
                    y + height > image_height * 0.25):
                regions.append(gray[y:y + height, x:x + width])
                candidate_regions.append((area, gray[y:y + height, x:x + width]))

        if candidate_regions:
            regions = [gray] + [region for _, region in sorted(
                candidate_regions, reverse=True
            )[:4]]

        best_text = ""
        for region in regions:
            if region.size == 0:
                continue
            # Upscaling and local thresholding substantially improve small plates.
            scale = max(2, min(5, 1200 // max(1, region.shape[1])))
            enlarged = cv2.resize(
                region, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )
            normalized = cv2.normalize(
                enlarged, None, 0, 255, cv2.NORM_MINMAX
            )
            variants = [
                cv2.threshold(
                    normalized, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1],
            ]
            for variant in variants:
                variant = cv2.morphologyEx(
                    variant, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)
                )
                text = pytesseract.image_to_string(
                    variant,
                    config=(
                        "--oem 3 --psm 7 "
                        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    )
                )
                cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
                if 3 <= len(cleaned) <= 12 and len(cleaned) > len(best_text):
                    best_text = cleaned
        return best_text
    except Exception as error:
        print(f"Tesseract OCR failed to run: {error}")
        return ""


while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = model.track(
        frame,
        classes=vehicle_classes,
        conf=0.7,
        imgsz=640,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # Crop and process each detected vehicle independently.
    boxes = results[0].boxes
    track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)

    for vehicle_index, (box, track_id) in enumerate(zip(boxes.xyxy, track_ids)):
        x1, y1, x2, y2 = map(int, box.tolist())
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        vehicle_crop = frame[y1:y2, x1:x2]
        if vehicle_crop.size == 0:
            continue

        # Running Tesseract on every vehicle in every frame causes most of the lag.
        cache_key = track_id if track_id is not None else f"untracked_{vehicle_index}"
        if (cache_key not in ocr_cache or
                frame_number - ocr_cache[cache_key][0] >= OCR_INTERVAL):
            detected_text = process_vehicle(vehicle_crop)
            ocr_cache[cache_key] = (frame_number, detected_text)
        else:
            detected_text = ocr_cache[cache_key][1]
        if detected_text:
            cv2.putText(
                vehicle_crop,
                detected_text.replace("\n", " "),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # Save the first labeled image for each tracked vehicle ID.
        if track_id is not None and track_id not in seen_vehicle_ids:
            image_path = os.path.join(
                saved_vehicle_dir,
                f"vehicle_{track_id}.jpg"
            )
            cv2.imwrite(image_path, vehicle_crop)
            seen_vehicle_ids.add(track_id)

    annotated_frame = results[0].plot()

    cv2.imshow("Vehicle Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()