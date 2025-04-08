import cv2
import pandas as pd
import os

# === File paths ===
video_path = r"C:\Users\sande\Downloads\Traffic.mp4"
yolo_path = 'evaluation/results/yolo_sort.txt'
v2_path = 'evaluation/results/v2_vehicle.txt'

# === Column setup (MOT format) ===
columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf']

# === Load predictions ===
yolo_df = pd.read_csv(yolo_path, header=None, names=columns)
v2_df = pd.read_csv(v2_path, header=None, names=columns)

# === Load video ===
cap = cv2.VideoCapture(video_path)
frame_id = 1

# === Target output size for display ===
target_width = 640
target_height = 360

# === Get original video size to scale bounding boxes ===
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale_x = target_width / original_width
scale_y = target_height / original_height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame_resized = cv2.resize(frame, (target_width, target_height))
    frame_yolo = frame_resized.copy()
    frame_v2 = frame_resized.copy()

    # Get frame-specific predictions
    yolo_f = yolo_df[yolo_df['frame'] == frame_id]
    v2_f = v2_df[v2_df['frame'] == frame_id]

    # === Draw YOLO+SORT predictions (green) ===
    for _, row in yolo_f.iterrows():
        x = int(row['x'] * scale_x)
        y = int(row['y'] * scale_y)
        w = int(row['w'] * scale_x)
        h = int(row['h'] * scale_y)
        cv2.rectangle(frame_yolo, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame_yolo, f'ID {int(row["id"])}', (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # === Draw V2 predictions (blue) ===
    for _, row in v2_f.iterrows():
        x = int(row['x'] * scale_x)
        y = int(row['y'] * scale_y)
        w = int(row['w'] * scale_x)
        h = int(row['h'] * scale_y)
        cv2.rectangle(frame_v2, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame_v2, f'ID {int(row["id"])}', (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # === Add labels ===
    cv2.putText(frame_yolo, "YOLO+SORT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame_v2, "V2 Tracker", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # === Concatenate and display ===
    combined = cv2.hconcat([frame_yolo, frame_v2])
    cv2.imshow("Side-by-Side Tracker Comparison", combined)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
