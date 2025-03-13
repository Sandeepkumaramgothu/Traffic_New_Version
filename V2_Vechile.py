import cv2
import numpy as np
import math

# Video source (update path if necessary)
cap = cv2.VideoCapture(r"C:\Users\sande\Downloads\Traffic.mp4")

# Minimum rectangle dimensions for a valid detection
min_width_react = 80
min_height_react = 80

# Counting line position and drawing parameters
count_line_position = 550
line_thickness = 3
offset = 6  # used for visualizing and can be adjusted if needed

# Initialize background subtractor
algo = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

def center_handle(x, y, w, h):
    """Calculate the center point of a bounding box."""
    x1 = int(w / 2)
    y1 = int(h / 2)
    return (x + x1, y + y1)

def distance(pt1, pt2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

# Dictionary to track vehicles
# Each key is a unique vehicle ID and value is a dict with current center, last center,
# a flag indicating if it has been counted, and frames missing.
vehicles = {}
next_vehicle_id = 0

# Counters for vehicles moving upward and downward
up_limit = 0
down_limit = 0

# Parameters for matching detections to existing vehicles
max_distance = 30         # Maximum distance to consider a detection the same vehicle
max_frames_missing = 5    # Remove track if not updated for these many frames

while True:
    ret, frame1 = cap.read()
    if not ret:
        print("Failed to read frame. Exiting...")
        break
    
    # Preprocess the frame: grayscale, blur, and background subtraction
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    
    # Find contours from the processed frame
    contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the counting line
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), line_thickness)
    
    # List to hold centers detected in the current frame
    detections = []
    
    # Process each contour and extract valid vehicle detections
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if w >= min_width_react and h >= min_height_react:
            center = center_handle(x, y, w, h)
            detections.append(center)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame1, center, 4, (0, 0, 255), -1)
            cv2.putText(frame1, "Vehicle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Keep track of vehicles updated in the current frame
    updated_vehicle_ids = set()
    
    # Attempt to match each detection to an existing tracked vehicle
    for det in detections:
        matched_vehicle_id = None
        for vehicle_id, vehicle in vehicles.items():
            if distance(det, vehicle['center']) < max_distance:
                matched_vehicle_id = vehicle_id
                break
        
        if matched_vehicle_id is not None:
            # Update the existing vehicle's information
            vehicles[matched_vehicle_id]['last_center'] = vehicles[matched_vehicle_id]['center']
            vehicles[matched_vehicle_id]['center'] = det
            vehicles[matched_vehicle_id]['frames_missing'] = 0
            updated_vehicle_ids.add(matched_vehicle_id)
        else:
            # Create a new vehicle track
            vehicles[next_vehicle_id] = {
                'center': det,
                'last_center': det,
                'counted': False,
                'frames_missing': 0
            }
            updated_vehicle_ids.add(next_vehicle_id)
            next_vehicle_id += 1
    
    # Increase frames_missing count for vehicles not updated this frame and remove stale tracks
    for vehicle_id in list(vehicles.keys()):
        if vehicle_id not in updated_vehicle_ids:
            vehicles[vehicle_id]['frames_missing'] += 1
        if vehicles[vehicle_id]['frames_missing'] > max_frames_missing:
            del vehicles[vehicle_id]
    
    # Check for line crossing for each tracked vehicle and update counters accordingly
    for vehicle_id, vehicle in vehicles.items():
        if not vehicle['counted']:
            last_y = vehicle['last_center'][1]
            current_y = vehicle['center'][1]
            # Vehicle moving downward (from above to below the counting line)
            if last_y < count_line_position and current_y >= count_line_position:
                down_limit += 1
                vehicle['counted'] = True
                # Visualize the crossing event
                cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), line_thickness)
                cv2.circle(frame1, vehicle['center'], 6, (0, 127, 255), -1)
            # Vehicle moving upward (from below to above the counting line)
            elif last_y > count_line_position and current_y <= count_line_position:
                up_limit += 1
                vehicle['counted'] = True
                cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), line_thickness)
                cv2.circle(frame1, vehicle['center'], 6, (0, 127, 255), -1)
    
    total_count = up_limit + down_limit
    # Display the counts on the frame
    cv2.putText(frame1, "Car Counter: " + str(total_count), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.putText(frame1, "Up: " + str(up_limit), (450, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.putText(frame1, "Down: " + str(down_limit), (450, 230), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    
    cv2.imshow("Video Original", frame1)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
