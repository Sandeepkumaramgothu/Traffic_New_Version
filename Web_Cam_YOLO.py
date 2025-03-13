from ultralytics import YOLO
import cv2
import numpy as np
import math
import cvzone


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r"C:\Users\sande\Downloads\Test_Video_3.mp4")
#model = YOLO("yolov8l.pt")
cap.set(3, 1280)
cap.set(4, 720)

class_names = ["person","bicycle","car","motobike","aeroplane","bus","train","truck","boat","traffic light",
"fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
"elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
"skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
"wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
"broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed",
"diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","micowave","oven",
"toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]
frame=0
while True:
    frame+=1
    print("Frame : ",frame)
    print("graphics check ",cv2.cuda.getCudaEnabledDeviceCount())
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("Using GPU")
    else:
        print("Not using GPU")
        
    success, img = cap.read()
    results = model(img,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1,y1), (x2,y2),(0, 255, 0), 3)
            #print(x1, y1, x2, y2)
            
            x,y,w,h=x1, y1, x2-x1, y2-y1
            #bbox = int(x), int(y), int(w), int(h)
            cvzone.cornerRect(img, (x, y, w, h))
            
            #cofidence
            conf = math.ceil((box.conf[0]*100))/100
            
            #class name
            cls = int(box.cls[0])
            
            cvzone.putTextRect(img, f'{class_names[cls]}  {conf}', (max(0,int(x)), max(35,int(y))),scale=1,thickness=2)
            #print("confidence : ",conf)
            #cv2.putText(img, f'{conf}', (max(0,int(x)), max(35,int(y))), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #image = results.render()
    #image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", img)
    cv2.waitKeyEx(1)
