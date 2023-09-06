from ultralytics import YOLO
import cv2
import cvzone
import math


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../YOLO_weights/yolov8m.pt")

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1,y1),(x2,y2),(255,0,255),3)

            w,h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1,y1,w,h))



    cv2.imshow("Image",img)
    cv2.waitKey(1)