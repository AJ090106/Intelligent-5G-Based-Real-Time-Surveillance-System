# Aman Mourya's Algorithm
# License Plate Recognition using YOLOv8 and OpenCV
from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture("input.mp4")

# COCO classes:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

vehicle_classes = [2, 3, 5, 7]

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(
        frame,
        classes=vehicle_classes,
        conf=0.4
    )

    annotated_frame = results[0].plot()

    cv2.imshow("Vehicle Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()