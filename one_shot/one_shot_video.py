from imutils.video import VideoStream
from imutils.video import FPS
from cv2 import VideoCapture,VideoWriter
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
from progressbar import ProgressBar,ETA
import keyboard

ap = argparse.ArgumentParser()
ap.add_argument("-v",help="Path to the video",required=True)
ap.add_argument("-o",help="Path to the output video",required=True)
ap.add_argument("-d",help="Show information abou processing",required=False,action="store_true")
ap.add_argument("-c",help="Minimum confidence to find face",required=False,type=float,default=0.7)
ap.add_argument("-t",help="Tolerance of distance of faces",required=False,type=float,default=0.6)
ap.add_argument("--model",help="Model to extract embeddings",required=False)
args = vars(ap.parse_args())

#Usar openCV para extrair embbedind do frame
if args.get("model") == None:
    opencv = False
else:
    opencv=True
    print("[INFO] - Loading embedding model")
    emb = cv2.dnn.readNetFromTorch(args["model"])

print("[INFO] - Loading face detector")
detector = cv2.dnn.readNetFromCaffe("models/face_detection_model/deploy.prototxt",
			"models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")


print("[INFO] - Loading known embeddings")
data = pickle.loads(open("known/embeddings.pickle","rb").read()) 
knownEmbeddings = []
knownNames = []
samples = {}
noDetected=0
frameEmb = np.empty((128,))
proba = 0

for e in data["embeddings"]:
	knownEmbeddings.append(e)

for n in data["names"]:
	knownNames.append(n)

samples = data["samples"]


print("[INFO] - Starting video read")
vc = VideoCapture(args["v"])
if vc.isOpened() == False:
	print("[ERROR] - Failed to open video")
	exit()

ret,frame = vc.read()
frame = imutils.resize(frame, width=600)

#Caso precise rodar o video
#frame = imutils.rotate(frame,angle=90)                

out = VideoWriter(args["o"]+".avi",cv2.VideoWriter_fourcc(*'XVID'),vc.get(cv2.CAP_PROP_FPS),(frame.shape[1],frame.shape[0]))

bar = ProgressBar(vc.get(cv2.CAP_PROP_FRAME_COUNT)+1).start()
bar.widgets.append(ETA())
count=1
pause = False

print("[INFO] - Processing video - Press 'p' to pause and 'q' to quit")
while vc.isOpened():

	ret,frame = vc.read()
	bar.update(count+1)
	count+=1
	if ret == True:
		frame = imutils.resize(frame, width=600,)
		#frame = imutils.rotate(frame,angle=90)
		(h, w) = frame.shape[:2]
			
		imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
		detector.setInput(imageBlob)
		detections = detector.forward()
		
		for i in range(0, detections.shape[2]):
		
			confidence = detections[0, 0, i, 2]
			if confidence > args["c"]:
				noDetected+=1
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]
				if fW > 250 or fH > 340:
					break
			
				if fW < 20 or fH < 20:
					continue
				
				if opencv:
					faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
					emb.setInput(faceBlob)
					frameEmb = emb.forward()		
				else:
					rgb = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
					encodings = face_recognition.face_encodings(rgb,model="cnn")
					for enc in encodings:
						frameEmb=enc
				
				
				
				matches = face_recognition.compare_faces(knownEmbeddings,frameEmb,tolerance=args["t"])
				name="Unknown"
				text = "{}".format(name)
				
				if True in matches:
					indexes = [i for (i,b) in enumerate(matches)if b]
					counts = {}
					for i in indexes:
						name =knownNames[i]
						counts[name] = counts.get(name,0) + 1
						
					name = max(counts,key=counts.get)
					conf =(counts[name]*100)/samples[name]
					proba = "{:.2f}%".format(conf)
					
					if conf/100 > 0.8: 
						text = "{} : {}".format(name, proba)
					else:
						name="Unknown"
						
					matchesConfidences = {}
					for d in counts.items():
						matchesConfidences[d[0]] = "{:.2f}%".format( (d[1]*100) / samples[d[0]])  
					
					if args["d"]:
						print("\nFrame#{}\nMatches = {}\n\nPredicted = {}\nConfidence = {}\n".format(count,matchesConfidences,name,proba))
					

				
				
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y),cv2.FONT_ITALIC,.45, (0, 0, 255), 2)
		


		text2="{} faces".format(noDetected)
		cv2.putText(frame,text2,(20,30),cv2.FONT_ITALIC,.60,(0,0,255),2)	
		noDetected=0
		out.write(frame)
		cv2.imshow("Output Preview",frame)
			
		key = cv2.waitKey(1) & 0xFF
		if key == ord("p"):
			pause=True
			print("Paused",end='\r')
			while pause or key == ord("p"):
				key = cv2.waitKey(1) & 0xFF
				if key == ord("p"):
					pause = False
		if key == ord("q"):
			print("Stopped by the user")
			vc.release()
			out.release()
			cv2.destroyAllWindows()
			bar.finish()
			exit()
		
	else:
		vc.release()
		out.release()
		bar.finish()
		cv2.destroyAllWindows()
		exit()


	 