from imutils.video import VideoStream
from imutils.video import FPS
from imutils.face_utils import FaceAligner
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import face_recognition
import dlib
import argparse
import keyboard

ap = argparse.ArgumentParser()
ap.add_argument("-c",help="minimum confidence to find face on the frame",required=False,type=float,default=0.8)
args = vars(ap.parse_args())

#Face detector, return the position of the faces in the image
print("[INFO] - Loading face detector")
detector = cv2.dnn.readNetFromCaffe("models/face_detection_model/deploy.prototxt",
			"models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")

sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(sp)



print("[INFO] - Starting video stream - Press 'p' to pause and 'q' to quit")
vs = VideoStream(src=0,resolution=(1280,720)).start()
time.sleep(2.0)

#Initializing variables
noDetected=0
count=1
fps = FPS().start()

while True:
	
	count+=1
	if count % 2 == 0:
		fps.update()
		noDetected=0
		
		frame = vs.read() # Read a frame
		frame = imutils.resize(frame, width=600) #Rezising to extract the Blob after
		(h, w) = frame.shape[:2] # Get the height and weight of the resized image

		imageBlob = cv2.dnn.blobFromImage( #Extract the blob of image to put in detector
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)


		detector.setInput(imageBlob) # Realize detection
		detections = detector.forward()

	
				
		for f in range(0, detections.shape[2]):# For each face detected

			confidence = detections[0, 0, f, 2]#Extract the confidence returned by the detector
			
			if confidence >= args["c"]: #Compare with the confidence passed by argument
				
				
				text = "{:.2f}%".format(confidence*100)
				noDetected+=1 #Faces detected in the frame counter
				box = detections[0, 0, f, 3:7] * np.array([w, h, w, h])# Convert the positions to a np.array
				(startX, startY, endX, endY) = box.astype("int")# Get the coordinates to cut the face from the frame
				
				face = frame[startY:endY, startX:endX] #Extract the face from the frame
				(fH, fW) = face.shape[:2]# Get the face height and weight			
				if fW > 250 or fH > 340 or fW < 20 or fH < 20:
					continue
				
				kp = sp(frame,dlib.rectangle(startX,startY,endX,endY))

				for k in range(0,kp.num_parts):
					pt =(kp.part(k).x,kp.part(k).y)
					cv2.line(frame,pt,pt,(0,0,255),6)



				if noDetected > 0:
					cv2.imshow("Face#{}".format(f),face)


				y = startY - 10 if startY - 10 > 10 else startY + 10	
				cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y),cv2.FONT_ITALIC,.45, (0, 00, 255), 2)
			
	text2="{} faces".format(noDetected)
	cv2.putText(frame,text2,(20,30),cv2.FONT_ITALIC,.60,(0,0,255),2)	
	cv2.imshow("Frame", frame)	
	
	key = cv2.waitKey(1) & 0xFF 
	
	if key == ord("q"):
		fps.stop()
		print("[INFO] - elapsed time: {:.2f}".format(fps.elapsed()))
		print("[INFO] - approx. FPS: {:.2f}".format(fps.fps()))
		vs.stop()
		cv2.destroyAllWindows()
		time.sleep(2)
		exit()
	if key == ord("p"):
			pause=True
			print("Paused",end='\r')
			while pause or key == ord("p"):
				key = cv2.waitKey(1) & 0xFF
				if key == ord("p"):
					pause = False
		

