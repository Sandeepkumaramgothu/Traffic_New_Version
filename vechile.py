import cv2
import numpy as np
import math

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r"C:\Users\sande\Downloads\Traffic.mp4")

min_width_react = 80
min_height_react = 80

count_line_position = 550
counter = 0
#Initialize Subtractor
algo = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

def center_handle(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx, cy

detect = []
offset = 6
counter = 0
up_limit = 0
down_limit = 0

while True:
    ret, frame1 = cap.read()
    if not ret:
        print("Failed to read frame. Exiting...")
        break
    
    #Apply Subtractor
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3,3), 5)
    #applying on the frame   
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255,127,0), 3)        
    
    for i, c in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_react) and (h >= min_height_react)   
        if not validate_counter:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Vehicle"+str(counter), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)    
        
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)
        
        if center[1] >= count_line_position-5 and center[1] <= count_line_position+5:
            counter += 1
            cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0,127,255), 3)
            cv2.line(frame1, (int(x+w/2), int(y+h/2)), (int(x+w/2), int(y+h/2)), (0,127,255), 3)
            
    for (x, y) in detect:
        if y < count_line_position - offset:
            up_limit += 1
        elif y > count_line_position + offset:
            down_limit += 1
                
    cv2.putText(frame1, "Car Counter: "+str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.putText(frame1, "Up: "+str(up_limit), (450, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)       
    cv2.putText(frame1, "Down: "+str(down_limit), (450, 230), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255,127,0), 3)    
    detect.clear()   
    print("Car Counter: "+str(counter))
    print("Up: "+str(up_limit))
    print("Down: "+str(down_limit)) 
    #cv2.imshow("Detector", dilatada)   // gery video frame
     
    #mask = algo.apply(frame1)
    #_, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    
    # Ensure the frame is not empty before displaying
    if frame1 is not None and frame1.size > 0:
        cv2.imshow("Video Original", frame1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()