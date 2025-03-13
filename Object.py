import cv2
from ultralytics import YOLO
import math
import numpy as np
import time


# Open video file or capture device
cap = cv2.VideoCapture(r"C:\Users\sande\Downloads\Test_Video.mp4")  # Replace 'road_video.mp4' with your video file

#model = YOLO("yolov8s.pt")  # Load YOLOv8s model

