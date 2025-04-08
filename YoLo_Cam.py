from ultralytics import YOLO
import cv2
import numpy as np
import cvzone
import os
from sort import Sort

# === Setup ===
os.makedirs("evaluation/results", exist_ok=True)
output_file = open("evaluation/results/yolo_sort.txt", "w")
frame_id = 1

# Load model
model = YOLO("yolov8l.pt")
class_names = model.names

# Load video
cap = cv2.VideoCapture(r"C:\Users\sande\Downloads\Traffic.mp4")
cap.set(3, 1280)
cap.set(4, 720)

# Initialize tracker
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

# Initialize line crossing counters
states = {}
up_count = 0
down_count = 0
line_position = 550

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            conf = round(box.conf[0].item(), 2)
            cls = int(box.cls[0])

            if cls >= len(class_names):
                continue

            label = class_names[cls]
            if label in ["car", "truck", "bus", "motorcycle"] and conf > 0.3:
                detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

    # Track updates
    tracks = tracker.update(detections)
    height, width, _ = img.shape
    line_color = (0, 0, 255)
    crossed_up, crossed_down = False, False

    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw box and ID
        cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), colorR=(255, 0, 255))
        cvzone.putTextRect(img, f"ID: {track_id}", (x1, y1 - 10), scale=1, thickness=2)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Line crossing logic
        current_state = "above" if cy < line_position else "below"
        if track_id not in states:
            states[track_id] = current_state
        else:
            old_state = states[track_id]
            if old_state == "above" and current_state == "below":
                down_count += 1
                crossed_down = True
            elif old_state == "below" and current_state == "above":
                up_count += 1
                crossed_up = True
            states[track_id] = current_state

        # Save to file (MOT format)
        w, h = x2 - x1, y2 - y1
        output_file.write(f"{frame_id},{track_id},{x1},{y1},{w},{h},1,-1,-1,-1\n")

    # Draw crossing line
    if crossed_up and crossed_down:
        line_color = (255, 0, 255)
    elif crossed_up:
        line_color = (255, 0, 0)
    elif crossed_down:
        line_color = (0, 255, 0)

    cv2.line(img, (0, line_position), (width, line_position), line_color, 3)
    cvzone.putTextRect(img, f"Up: {up_count}", (10, 50), scale=2, thickness=2)
    cvzone.putTextRect(img, f"Down: {down_count}", (10, 110), scale=2, thickness=2)

    frame_id += 1
    cv2.imshow("YOLO+SORT Tracker", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
output_file.close()
print("✅ Tracking results saved to:", "evaluation/results/yolo_sort.txt")
print("✅ YOLO+SORT tracking completed.")   