from ultralytics import YOLO
import cv2

#model = YOLO("yolov5l.pt")  # Load YOLOv8s model

# Load image from file
results = model(r"C:\Users\sande\OneDrive\Pictures\YOLO\1672.jpeg",show=True,scale=0.5)
cv2.waitKey(0)
