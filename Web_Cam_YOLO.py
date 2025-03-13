from ultralytics import YOLO
import cv2
import numpy as np
import cvzone
from sort import Sort  # or from sort import *

# --------------------------
# 1. VIDEO SETUP
# --------------------------
cap = cv2.VideoCapture(r"C:\Users\sande\Downloads\Test_Video_3.mp4")
cap.set(3, 1280)
cap.set(4, 720)

# --------------------------
# 2. INSTANTIATE YOLO
#    - conf=0.5: show detections above 50% confidence
#    - iou=0.3 : lower IoU threshold to reduce merging
# --------------------------
model = YOLO("yolov8l.pt")
model.overrides['conf'] = 0.5
model.overrides['iou'] = 0.3

# --------------------------
# 3. CLASS NAMES
# --------------------------
class_names = [
    "person", "bicycle", "car", "motorcycle", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# --------------------------
# 4. CREATE SORT TRACKER
# --------------------------
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

# --------------------------
# 5. CROSSING LOGIC SETUP
#    We track each vehicle's last known "above" or "below" state
# --------------------------
states = {}       # track_id -> "above" or "below"
up_count = 0      # vehicles going from below -> above
down_count = 0    # vehicles going from above -> below
line_position = 550

frame = 0
while True:
    frame += 1
    success, img = cap.read()
    if not success:
        break

    # ----------------------
    # YOLO Inference
    # ----------------------
    results = model(img, stream=True)
    
    # Prepare an empty array for the current detections [x1, y1, x2, y2, conf]
    detections = np.empty((0, 5))

    # ----------------------
    # 6. Parse YOLO Results
    # ----------------------
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            conf = round(box.conf[0].item(), 2)
            cls = int(box.cls[0])
            current_class = class_names[cls]

            # Focus only on certain vehicle classes
            if current_class in ["car", "truck", "bus", "motorcycle"] and conf > 0.3:
                # Add detection to array: [x1, y1, x2, y2, conf]
                detection = [x1, y1, x2, y2, conf]
                detections = np.vstack((detections, detection))

                # Optional: Draw detection bounding box and label
                cvzone.putTextRect(
                    img,
                    f"{current_class} {conf}",
                    (max(0, x1), max(35, y1)),
                    scale=1,
                    thickness=2,
                    offset=3
                )

    # ----------------------
    # 7. SORT Tracker Update
    # ----------------------
    tracks = tracker.update(detections)

    # By default, let's draw the line in red
    line_color = (0, 0, 255)

    # Flags to see if any crossing occurred this frame
    crossed_up = False
    crossed_down = False

    # ----------------------
    # 8. Draw Tracker Boxes/IDs + Detect Crossing
    # ----------------------
    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)

        # Draw bounding box from tracker
        cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), colorR=(255, 0, 255))
        cvzone.putTextRect(img, f"ID: {track_id}", (x1, y1 - 10), scale=1, thickness=2)
        
        # Center of the tracked bounding box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Determine current state (above or below the line)
        current_state = "above" if cy < line_position else "below"
        
        # If this track_id is new, record its initial state
        if track_id not in states:
            states[track_id] = current_state
        else:
            old_state = states[track_id]
            # If the state changed from above -> below, increment down_count
            if old_state == "above" and current_state == "below":
                down_count += 1
                crossed_down = True
            # If the state changed from below -> above, increment up_count
            elif old_state == "below" and current_state == "above":
                up_count += 1
                crossed_up = True
            
            # Update the stored state
            states[track_id] = current_state

    # ----------------------
    # 9. Color the line based on any crossing
    # ----------------------
    if crossed_up and crossed_down:
        line_color = (255, 0, 255)  # magenta if both up & down happen in same frame
    elif crossed_up:
        line_color = (255, 0, 0)    # blue if there's an up crossing
    elif crossed_down:
        line_color = (0, 255, 0)    # green if there's a down crossing
    
    # Draw the reference line
    cv2.line(img, (0, line_position), (1280, line_position), line_color, 3)

    # ----------------------
    # 10. Show Up/Down Counts
    # ----------------------
    cvzone.putTextRect(img, f"Up: {up_count}", (10, 50), scale=2, thickness=2)
    cvzone.putTextRect(img, f"Down: {down_count}", (10, 110), scale=2, thickness=2)

    # Display the frame
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
