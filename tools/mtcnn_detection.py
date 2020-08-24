from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from mtcnn.mtcnn import MTCNN
name="Face"
print("[INFO] - Loading face detector")
detector = MTCNN()
print("[INFO] starting video stream")
vs = VideoStream(src=0).start()
time.sleep(2.0)

fps = FPS().start()
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]
	
	faces=detector.detect_faces(frame)
	for face in faces:

		confidence = face['confidence']
	
		if confidence > 0.55:
					
			x,y,w,h =face['box']
			
			text = "{}: {:.2f}%".format(name, confidence*100)
			kp = face['keypoints']
			

			cv2.line(frame,kp['left_eye'],kp['left_eye'],(0,0,255),6)
			cv2.line(frame,kp['right_eye'],kp['right_eye'],(0,0,255),6)
			
			cv2.line(frame,kp['mouth_left'],kp['mouth_left'],(0,0,255),6)
			cv2.line(frame,kp['mouth_right'],kp['mouth_right'],(0,0,255),6)
			
			cv2.line(frame,kp['nose'],kp['nose'],(0,0,255),6)
			

			cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 0, 255), 1)
			
			cv2.putText(frame, text, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		
	fps.update()
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	
	if key == ord("q"):
		break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
time.sleep(2.0)