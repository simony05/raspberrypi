import cv2
import time
from picamera2 import Picamera2

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

model = 'model.tflite'
num_threads = 4

dispW = 640
dispH = 360

picam2 = Picamera2()
picam2.preview_configuration.main.size = (dispW, dispH)
picam2.preview_configuration.main.format = 'RGB888'
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
cam.set(cv2.CAP_PROP_FPS, 30)

pos = (20, 60)
font = cv2.FONT_HERSHEY_SIMPLEX
height = 1.5
weight = 3
myColor = (255, 0, 0)

fps = 0

base_options = core.BaseOptions(file_name = model, use_coral = False, num_threads = num_threads)
detection_options = processor.DetectionOptions(max_results = 4, score_threshold = 0.3)
options = vision.ObjectDetectorOptions(base_options = base_options, detection_options = detection_options)
detector = vision.ObjectDetector.create_from_options(options)
tStart = time.time()

while True:
	ret, im = cam.read()
	#im = picam2.capture.array()
	#im = cv2.flip(im, -1)
	imRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	#imTensor = vision.TensorImage.create_from_array(imRGB)
	#detections = detector.detect(imTensor)
	#image = utils.visualize(im, detections)
	cv2.putText(im, str(int(fps)) + ' FPS', pos, font, height, myColor, weight)
	cv2.imshow('Camera', im)
	if cv2.waitKey(1) == ord('q'):
		break
	tEnd = time.time()
	loopTime = tEnd - tStart
	fps = 0.9 * fps + .1 * 1 / loopTime
cv2.destroyAllWindows()

