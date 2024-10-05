import cv2
from picamera2 import Picamera2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np

picam = Picamera2()
picam.preview_configuration.main.size = (640, 480)
picam.preview_configuration.main.format = "RGB888"
picam.preview_configuration.align()
picam.start()

model = YOLO("yolov8 best.pt")

class_list = ["circle", "square", "triangle"]
count = 0

while True:
	img = picam.capture_array()
	
	count += 1
	if count % 3 != 0:
		continue
	img = cv2.flip(img, -1)
	results = model.predict(img)
	a = results[0].boxes.data
	px = pd.DataFrame(a).astype("float")
	
	for index, row in px.iterrows():
		x1 = int(row[0])
		y1 = int(row[1])
		x2 = int(row[2])
		y2 = int(row[3])
		d = int(row[5])
		c = class_list[d]
		
		cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
		cvzone.putTextRect(img, f'{c}', (x1, y1), 1, 1)
	
	cv2.imshow("Camera", img)
	
	if cv2.waitKey(1) == ord('q'):
		break

cv2.destroyAllWindows()
