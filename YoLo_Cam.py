from ultralytics import YOLO
import cv2
import numpy as np
import cvzone
from sort import Sort

cap = cv2.VideoCapture(r"C:\Users\sande\Downloads\Test_Video_3.mp4")
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("yolov8l.pt")
model.overrides['conf'] = 0.5
model.overrides['iou'] = 0.3

class_names = ["person", "bicycle", "car", "motorcycle", "aeroplane", "bus",
               "train", "truck", "boat", "traffic light", "fire hydrant",
               "stop sign", "parking meter", "bench", "bird", "cat", "dog",
               "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
               "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
               "skis", "snowboard", "sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket",
               "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
               "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
               "hot dog", "pizza", "donut", "cake", "chair", "sofa",
               "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
               "laptop", "mouse", "remote", "keyboard", "cell phone",
               "microwave", "oven", "toaster", "sink", "refrigerator", "book",
               "clock", "vase", "scissors", "teddy bear", "hair drier",
               "toothbrush"]

tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

# Track states: ID -> "above" or "below"
states = {}
up_count = 0
down_count = 0
line_position = 550

while True:
    success, img = cap.read()
    if not success:
        break

    # YOLO inference
    results = model(img, stream=True)
    
    # Prepare array for detections: [x1, y1, x2, y2, conf]
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            conf = round(box.conf[0].item(), 2)
            cls = int(box.cls[0])
            current_class = class_names[cls]

            if current_class in ["car", "truck", "bus", "motorcycle"] and conf > 0.3:
                detection = [x1, y1, x2, y2, conf]
                detections = np.vstack((detections, detection))

    # Update tracker
    tracks = tracker.update(detections)

    # Determine the image width for drawing a full-width line
    height, width, _ = img.shape
    line_color = (0, 0, 255)  # Default to red

    # Flags to see if any crossing happened this frame
    crossed_up = False
    crossed_down = False

    # Draw tracker boxes and do up/down logic
    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        
        # Draw bounding box
        cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), colorR=(255, 0, 255))
        cvzone.putTextRect(img, f"ID: {track_id}", (x1, y1 - 10), scale=1, thickness=2)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # above/below line
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

    # Change line color based on crossing
    if crossed_up and crossed_down:
        line_color = (255, 0, 255)  # magenta
    elif crossed_up:
        line_color = (255, 0, 0)    # blue
    elif crossed_down:
        line_color = (0, 255, 0)    # green

    # Draw a full-width line at line_position
    cv2.line(img, (0, line_position), (width, line_position), line_color, 3)

    # Show up/down counts
    cvzone.putTextRect(img, f"Up: {up_count}", (10, 50), scale=2, thickness=2)
    cvzone.putTextRect(img, f"Down: {down_count}", (10, 110), scale=2, thickness=2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
