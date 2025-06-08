import torch

from ultralytics import YOLO

# Load model with different method
model = YOLO()
model.load('yolov8_model.pt')
