from imutils.video import VideoStream
from imutils.video import FPS
from imutils.face_utils import FaceAligner
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
ap.add_argument("-d",help="Show information about processing",required=False,action="store_true")
ap.add_argument("-c",help="Minimum confidence to find face",required=False,type=float,default=0.8)
ap.add_argument("-p",help="Minimum confidence to predict a person, matches in dataset",required=False,type=float,default=0.6)
ap.add_argument("--dlib",help="Use dlib's model to extract embeddings",required=False,action="store_true")
args = vars(ap.parse_args())

#Usar openCV para extrair embbedind do frame
if args["dlib"]:
    print("[INFO] - Using DLIB embedding model")
    opencv = False
else:
    opencv=True
    print("[INFO] - Loading OpenCV embedding model")
    emb = cv2.dnn.readNetFromTorch("models/nn4.small1.v1.t7")

print("[INFO] - Loading face detector")
detector = cv2.dnn.readNetFromCaffe("models/face_detection_model/deploy.prototxt",
			"models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")

sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(sp)


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

fps = FPS().start()
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
		name="Unknown"
		text = "{}".format(name)
		frame = imutils.resize(frame, width=600)
		#frame = imutils.rotate(frame,angle=90)
		(h, w) = frame.shape[:2]
		frameOut = np.copy(frame)
		
		imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
		detector.setInput(imageBlob)
		detections = detector.forward()
		
		for f in range(0, detections.shape[2]):
		
			confidence = detections[0, 0, f, 2]
			if confidence > args["c"]:
				noDetected+=1
				box = detections[0, 0, f, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				face = frame[startY:endY, startX:endX]
				
				al = np.copy(frame)
				gray=cv2.cvtColor(al,cv2.COLOR_BGR2GRAY)
				
				face = fa.align(al,
				gray,
				dlib.rectangle(startX,startY,endX,endY))		
				
				
				# (fH, fW) = face.shape[:2]
				# if fW > 250 or fH > 340:
				# 	break
			
				# if fW < 20 or fH < 20:
				# 	continue

				if noDetected > 0:
					cv2.imshow("Face#{}".format(f),face)
				
				frameEmb=np.empty(128,)
				if opencv:
					faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
					emb.setInput(faceBlob)
					frameEmb = emb.forward()
					frameEmb = frameEmb.flatten()		
				else:
					rgb = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
					encodings=[]
					encodings = face_recognition.face_encodings(rgb,num_jitters=2,model="large")
					for enc in encodings:
						frameEmb=enc
				
				
			#Compare the face embedding of te frame with all faces registered on the dataset
				distances=np.empty(len(knownEmbeddings),)
				distances = face_recognition.face_distance(knownEmbeddings,frameEmb)
				
				faceDistances={}
				for (i,d) in enumerate(distances):
					n = knownNames[i]
					faceDistances[n] = faceDistances.get(n,max(distances))
					if d < faceDistances[n]: 
						faceDistances[n]=d

			
				name = min(faceDistances,key=faceDistances.get) # Get the name of the face with the minimum distance
				distance = faceDistances.get(name)
				accuracy = max(distances) - distance
				if distance <= args["p"]: 
					text = "#{}-{} : {:.2f}%".format(f,name, accuracy*100)
				else:
					name="Unknown"

				print(faceDistances)
				if args["d"]:
						print("\nFace#{}\nLooks like = {}\nPredicted = {}\nDistance = {}\nAccuracy = {:.2f}%\n".format(f,min(faceDistances,key=faceDistances.get),name,distance,accuracy*100))
				
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frameOut, (startX, startY), (endX, endY),(0, 0, 255), 1)
				cv2.putText(frameOut, text, (startX, y),cv2.FONT_ITALIC,.45, (0, 0, 255), 1)
		


		text2="{} faces".format(noDetected)
		cv2.putText(frameOut,text2,(20,30),cv2.FONT_ITALIC,.60,(0,0,255),2)	
		
		noDetected=0
		out.write(frameOut)
		cv2.imshow("Output Preview",frameOut)		
		fps.update()

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
			fps.stop()
			print("[INFO] - elapsed time: {:.2f}".format(fps.elapsed()))
			print("[INFO] - approx. FPS: {:.2f}".format(fps.fps()))
			vc.release()
			out.release()
			cv2.destroyAllWindows()
			bar.finish()
			time.sleep(2)
			exit()
		
	else:
		fps.stop()
		print("[INFO] - elapsed time: {:.2f}".format(fps.elapsed()))
		print("[INFO] - approx. FPS: {:.2f}".format(fps.fps()))	
		vc.release()
		out.release()
		bar.finish()
		cv2.destroyAllWindows()
		time.sleep(2)
		exit()



	 