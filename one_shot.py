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
import math


ap = argparse.ArgumentParser()
ap.add_argument("-d",help="show information about processing",required=False,action="store_true")
ap.add_argument("-c",help="minimum confidence to find face on the frame",required=False,type=float,default=0.8)
ap.add_argument("-p",help="minimum confidence to predict a person, matches in dataset",required=False,type=float,default=0.55)
ap.add_argument("-t",help="tolerance of distance",required=False,type=float,default=0.7)
ap.add_argument("--opencv",help="use opencv model to extract embeddings",required=False,action="store_true")

args = vars(ap.parse_args())



def distance2conf(face_distance,tolerance):
	if face_distance > tolerance:
		range = (1.0 - tolerance)
		linear_val = (1.0 - face_distance) / (range * 2.0)
		return linear_val
	else:
		range = tolerance
		linear_val = 1.0 - (face_distance / (range * 2.0))
		return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) *2,0.2))


if args["opencv"]:	
#Use openCV model to extract embeddings
    opencv=True
    print("[INFO] - Using OpenCV embedding model")
    emb = cv2.dnn.readNetFromTorch("models/nn4.small1.v1.t7")

else:
	#Use dlib embeddings extractor
    print("[INFO] - Using DLIB embedding model")
    opencv = False


#Face detector, return the position of the faces in the image
print("[INFO] - Loading face detector")
detector = cv2.dnn.readNetFromCaffe("models/face_detection_model/deploy.prototxt",
			"models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")


sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(sp)

# Dataset with the embbedings knowned, frames will be compared with each face here
print("[INFO] - Loading known embeddings")
data = pickle.loads(open("known/embeddings.pickle","rb").read()) 
knownEmbeddings = []
knownNames = []
facePaths = []

#Loading to variables
for e in data["embeddings"]:
	knownEmbeddings.append(e)

for n in data["names"]:
	knownNames.append(n)

#facePaths for each person in the database
for fp in data["facePaths"]:
	facePaths.append(fp)


print("[INFO] - Starting video stream - Press 'p' to pause and 'q' to quit")
vs = VideoStream(src=0,resolution=(1280,720)).start()
time.sleep(2.0)


#Initializing variables
count=1
fps = FPS().start()

while True:
	
	
	noDetected=0
	frame = vs.read() # Read a frame
	frame = imutils.resize(frame, width=600) #Rezising to extract the Blob after
	(h, w) = frame.shape[:2] # Get the height and weight of the resized image
	frameOut =  np.copy(frame)
	count+=1
	if count %2 == 0:
		fps.update()
		imageBlob = cv2.dnn.blobFromImage( #Extract the blob of image to put in detector
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)
		detector.setInput(imageBlob) # Realize detection
		detections = detector.forward()

		for f in range(0, detections.shape[2]):# For each face detected
			name="Unknown"
			text = "{}".format(name)
			confidence = detections[0, 0, f, 2]#Extract the confidence returned by the detector
			if confidence >= 0.9: #Compare with the confidence passed by argument
				noDetected+=1 #Faces detected in the frame counter
				box = detections[0, 0, f, 3:7] * np.array([w, h, w, h])# Convert the positions to a np.array
				(startX, startY, endX, endY) = box.astype("int")# Get the coordinates to cut the face from the frame
				boxFace = frame[startY:endY, startX:endX] #Extract the face from the frame
				(fH, fW) = boxFace.shape[:2]# Get the face height and weight			
				if fW > 250 or fH > 340 or fW < 20 or fH < 20:
					continue
				
				al = np.copy(frame)
				gray=cv2.cvtColor(al,cv2.COLOR_BGR2GRAY)
				face = fa.align(al,
				gray,
				dlib.rectangle(startX,startY,endX,endY))
				if noDetected > 0:
					cv2.imshow("Face#{}".format(f),face)
				frameEmb = np.empty(128,)
				if opencv:#Using openCV to extract the frame embeddings
					faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
					emb.setInput(faceBlob)
					frameEmb = emb.forward()		
					frameEmb = frameEmb.flatten()
				else:#Using dlib to extract the embeddings
					rgb = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
					locations = face_recognition.face_locations(rgb,model="cnn")
					encodings = face_recognition.face_encodings(rgb,locations,num_jitters=2,model="large")
					for enc in encodings:
						frameEmb=enc
				#Compare the face embedding of te frame with all faces registered on the dataset
				distances=np.empty(len(knownEmbeddings),)
				distances = face_recognition.face_distance(knownEmbeddings,frameEmb)
				faceDistances={}
				matchCount={}
				matchInfo={}
				for (i,d) in enumerate(distances):
					if d <= args["t"]:
						n = knownNames[i]
						matchCount[n] = matchCount.get(n,0)+1
						matchInfo[n+"distance"] = faceDistances.get(i,max(distances))
						if matchInfo.get(n+"distance",0) > d:
							matchInfo[n+"index"] = i;							
							matchInfo[n+"distance"] = d
					faceDistances[i] = faceDistances.get(i,max(distances))
					if d < faceDistances[i]: 
						faceDistances[i]=d
				ind = min(faceDistances,key=faceDistances.get) # Get the name with minimum distance
				distance = faceDistances.get(ind)
				
				if len(matchCount) > 0:
					matchName = max(matchCount,key=matchCount.get)
					matchInd = matchInfo.get(matchName+"index")
					matchDis = matchInfo.get(matchName+"distance")
					nOfMatches = (matchCount.get(matchName))
					if matchDis <= distance and nOfMatches > 1:
						ind = matchInd
						name = knownNames[ind]
						distance = matchDis - (nOfMatches/100)
						
				probability = distance2conf(distance,args["t"])
				name = knownNames[ind]				
				faceComparedPath = facePaths[ind]

				if probability >= args["p"] : 
					text = "#{}-{} : {:.2f}%".format(f,name, probability*100)
				else:
					name="Unknown"
				faceCompared = cv2.imread(faceComparedPath)
				imutils.resize(faceCompared,width=600,height=600)
				cv2.imshow("Face#{} Best match".format(f),faceCompared)

				if args["d"]:
						print("\nFace#{}\nLooks like = {}\nPredicted = {}\nDistance = {}\nProbability = {:.2f}%\nMatch count = {}\n".format(f,knownNames[ind],name,distance,probability*100,matchCount.get(name,-1)))
				y = startY - 10 if startY - 10 > 10 else startY + 10	
				cv2.rectangle(frameOut, (startX, startY), (endX, endY),(0, 0, 255), 2)
				cv2.putText(frameOut, text, (startX, y),cv2.FONT_ITALIC,.45, (0, 255, 255), 2)
		
				text2="{} faces".format(noDetected)
				cv2.putText(frameOut,text2,(20,30),cv2.FONT_ITALIC,.60,(0,0,255),2)	
				cv2.imshow("Output", frameOut)	
	
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
		

