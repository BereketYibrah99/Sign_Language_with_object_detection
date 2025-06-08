from ultralytics import YOLO
import torch
model = YOLO('/Users/bereket/Desktop/Python/YOLOcustom/runs/detect/train7/weights/best.pt')
results = model.train(data='data.yaml', epochs=10)

# Check if ckpt is None
if model.ckpt is None:
    print("Checkpoint is None! Training might have failed.")
else:
    model.save('SignDetModel')