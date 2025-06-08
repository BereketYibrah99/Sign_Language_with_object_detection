import cv2
from ultralytics import YOLO
import cvzone
import torch


# Load YOLOv8 model from local path
model = YOLO('signDecModel.pt')  # Update this path to where you saved the .pt file

# Get the names of the classes
class_names = model.names

cap = cv2.VideoCapture(0)

while True:
    sucess, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,200),2)
            cv2.putText(img,f'{class_names[cls]}',(x1,y1),2,1,(0,255,160),2)
            #img = img[y1:y2,x1:x2]
    cv2.imshow("Image",img)
    cv2.waitKey(1)