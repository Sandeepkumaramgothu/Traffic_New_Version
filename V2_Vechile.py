import cv2
import numpy as np
import math
import os

# === Paths ===
video_path = r"C:\Users\sande\Downloads\Traffic.mp4"
output_path = "evaluation/results/v2_vehicle.txt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
output_file = open(output_path, "w")

# === Video Source ===
cap = cv2.VideoCapture(video_path)
frame_id = 1

# === Parameters ===
min_width_react = 80
min_height_react = 80
count_line_position = 550
line_thickness = 3
offset = 6

# Background subtractor
algo = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Vehicle tracking state
vehicles = {}
next_vehicle_id = 0
up_limit = 0
down_limit = 0
max_distance = 30
max_frames_missing = 5

def center_handle(x, y, w, h):
    return (x + w // 2, y + h // 2)

def distance(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

# === Main Loop ===
while True:
    ret, frame1 = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break

    # === Preprocessing ===
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    fg_mask = algo.apply(blur)
    dilated = cv2.dilate(fg_mask, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    processed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    # === Contour Detection ===
    contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), line_thickness)

    detections = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= min_width_react and h >= min_height_react:
            center = center_handle(x, y, w, h)
            detections.append((center, x, y, w, h))
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame1, center, 4, (0, 0, 255), -1)
            cv2.putText(frame1, "Vehicle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # === Vehicle Tracking Logic ===
    updated_vehicle_ids = set()

    for center, x, y, w, h in detections:
        matched_vehicle_id = None
        for vehicle_id, vehicle in vehicles.items():
            if distance(center, vehicle['center']) < max_distance:
                matched_vehicle_id = vehicle_id
                break

        if matched_vehicle_id is not None:
            vehicles[matched_vehicle_id]['last_center'] = vehicles[matched_vehicle_id]['center']
            vehicles[matched_vehicle_id]['center'] = center
            vehicles[matched_vehicle_id]['frames_missing'] = 0
            vehicles[matched_vehicle_id]['bbox'] = (x, y, w, h)
            updated_vehicle_ids.add(matched_vehicle_id)
        else:
            vehicles[next_vehicle_id] = {
                'center': center,
                'last_center': center,
                'bbox': (x, y, w, h),
                'counted': False,
                'frames_missing': 0
            }
            updated_vehicle_ids.add(next_vehicle_id)
            next_vehicle_id += 1

    # Remove lost tracks
    for vehicle_id in list(vehicles.keys()):
        if vehicle_id not in updated_vehicle_ids:
            vehicles[vehicle_id]['frames_missing'] += 1
        if vehicles[vehicle_id]['frames_missing'] > max_frames_missing:
            del vehicles[vehicle_id]

    # === Count Line Crossing ===
    for vehicle_id, vehicle in vehicles.items():
        if not vehicle['counted']:
            last_y = vehicle['last_center'][1]
            current_y = vehicle['center'][1]
            if last_y < count_line_position and current_y >= count_line_position:
                down_limit += 1
                vehicle['counted'] = True
                cv2.circle(frame1, vehicle['center'], 6, (0, 127, 255), -1)
            elif last_y > count_line_position and current_y <= count_line_position:
                up_limit += 1
                vehicle['counted'] = True
                cv2.circle(frame1, vehicle['center'], 6, (0, 127, 255), -1)

    # === Display Stats ===
    total_count = up_limit + down_limit
    cv2.putText(frame1, f"Car Counter: {total_count}", (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.putText(frame1, f"Up: {up_limit}", (450, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.putText(frame1, f"Down: {down_limit}", (450, 230), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # === Save to MOT format ===
    for vehicle_id, vehicle in vehicles.items():
        x, y, w, h = vehicle['bbox']
        output_file.write(f"{frame_id},{vehicle_id},{x},{y},{w},{h},1,-1,-1,-1\n")

    cv2.imshow("V2 Vehicle Tracker", frame1)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_id += 1

# === Cleanup ===
cap.release()
output_file.close()
cv2.destroyAllWindows()
print("✅ Tracking results saved to:", output_path)
print("✅ Vehicle tracking completed.")