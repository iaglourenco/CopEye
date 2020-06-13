from cv2 import VideoCapture,VideoWriter
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from progressbar import ProgressBar,ETA




ap = argparse.ArgumentParser()
ap.add_argument("-v",help="Path to the video",required=True,type=str)
args = vars(ap.parse_args())

name=""
print("[INFO] - Loading face detector")
detector = cv2.dnn.readNetFromCaffe("models/face_detection_model/deploy.prototxt",
			"models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")

print("[INFO] - Loading embedding model")
emb = cv2.dnn.readNetFromTorch("models/nn4.v2.t7")

recognizer = pickle.loads(open("pickle/recog.pickle", "rb").read())
le = pickle.loads(open("pickle/le.pickle", "rb").read())

print("[INFO]- Starting VideoStream")
vc = VideoCapture(args["v"])
if vc.isOpened() == False:
	print("[ERROR] - Failed to open video")
	exit()

ret,frame = vc.read()
frame = imutils.resize(frame, width=600)

#Caso precise rodar o video
#frame = imutils.rotate(frame,angle=90)                

out = VideoWriter("SVM-output.avi",cv2.VideoWriter_fourcc(*'XVID'),vc.get(cv2.CAP_PROP_FPS),(frame.shape[1],frame.shape[0]))

bar = ProgressBar(vc.get(cv2.CAP_PROP_FRAME_COUNT)+1).start()
bar.widgets.append(ETA())
count=1
pause = False

noDetected=0
while True:

	ret,frame = vc.read()
	bar.update(count+1)
	count+=1
	if ret ==True:
		frame = imutils.resize(frame, width=600)
		#frame = imutils.rotate(frame,angle=90)
		(h, w) = frame.shape[:2]
	
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)
	
		detector.setInput(imageBlob)
		detections = detector.forward()

		for i in range(0, detections.shape[2]):

			confidence = detections[0, 0, i, 2]
	
			if confidence > 0.8:
				noDetected+=1
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
	
				(startX, startY, endX, endY) = box.astype("int")
	
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]
				if fW > 250 or fH > 340:
					break
				
				if fW < 20 or fH < 20:
					continue

				
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
				emb.setInput(faceBlob)
				frameEmb = emb.forward()		
	
				preds = recognizer.predict_proba(frameEmb)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]
				if proba < 0.5:
					name = "Unknown"

				text = "{}: {:.2f}%".format(name, proba * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y),
					cv2.FONT_ITALIC,.45, (0, 0, 255), 2)
		text2="{} faces".format(noDetected)
		cv2.putText(frame,text2,(20,30),cv2.FONT_ITALIC,.60,(0,0,255),2)	
		noDetected=0
		out.write(frame)
		cv2.imshow("Output Preview", frame)
		key = cv2.waitKey(1) & 0xFF
	
		if key == ord("q"):
			vc.release()
			out.release()
			bar.finish()
			cv2.destroyAllWindows()
			exit()
		if key == ord("p"):
			pause=True
			print("Paused",end='\r')
			while pause or key == ord("p"):
				key = cv2.waitKey(1) & 0xFF
				if key == ord("p"):
					pause = False
	else:
		vc.release()
		out.release()
		bar.finish()
		cv2.destroyAllWindows()
		exit()

