import cv2
from ultralytics import YOLO
import torch

model = YOLO("best.pt")
device = 0 if torch.cuda.is_available() else 'cpu'

cap = cv2.VideoCapture("http://192.168.100.231:8080/video")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    results = model(frame, device=device)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Webcam Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()