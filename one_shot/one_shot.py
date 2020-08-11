from imutils.video import VideoStream
from imutils.video import FPS
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
ap.add_argument("-d",help="Show information about processing",required=False,action="store_true")
ap.add_argument("-c",help="Minimum confidence to find face",required=False,type=float,default=0.8)
ap.add_argument("-t",help="Tolerance of distance of faces",required=False,type=float,default=0.6)
ap.add_argument("-p",help="Minimum confidence to predict a person, matches in dataset",required=False,type=float,default=0.6)
ap.add_argument("--dlib",help="Model to extract embeddings",required=False)

args = vars(ap.parse_args())

if args.get("dlib") == None:
	#Use dlib embeddings extractor
    opencv = False
else:
#Use openCV model to extract embeddings
    opencv=True
    print("[INFO] - Loading embedding model")
    emb = cv2.dnn.readNetFromTorch(args["model"])


#Face detector, return the position of the faces in the image
print("[INFO] - Loading face detector")
detector = cv2.dnn.readNetFromCaffe("models/face_detection_model/deploy.prototxt",
			"models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")

# Dataset with the embbedings knowned, frames will be compared with each face here
print("[INFO] - Loading known embeddings")
data = pickle.loads(open("known/embeddings.pickle","rb").read()) 
knownEmbeddings = []
knownNames = []
samples = {}

#Loading to variables
for e in data["embeddings"]:
	knownEmbeddings.append(e)

for n in data["names"]:
	knownNames.append(n)

#Number of face samples for each person in the database
samples = data["samples"]


print("[INFO] - Starting video stream - Press 'p' to pause and 'q' to quit")
vs = VideoStream(src=0,resolution=(1920,1080)).start()
time.sleep(2.0)

fps = FPS().start()

#Initializing variables
noDetected=0
frameEmb = np.empty((128,))
proba = 0
count = 0
name="Unknown"
text = "{}".format(name)
while True:

	frame = vs.read() # Read a frame
	count+=1 #Frame counter
	# frame = imutils.resize(frame, width=600) #Rezising to extract the Blob after
	(h, w) = frame.shape[:2] # Get the height and weight of the resized image
	
	imageBlob = cv2.dnn.blobFromImage( #Extract the blob of image to put in detector
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
	
	
	detector.setInput(imageBlob) # Realize detection
	detections = detector.forward()
	
	for i in range(0, detections.shape[2]):# For each face detected
				
		confidence = detections[0, 0, i, 2]#Extract the confidence returned by the detector
		if confidence > args["c"]: #Compare with the confidence passed by argument
			noDetected+=1 #Faces detected in the frame counter
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])# Convert the positions to a np.array
			(startX, startY, endX, endY) = box.astype("int")# Get the coordinates to cut the face from the frame
			face = frame[startY:endY, startX:endX] #Extract the face from the frame
			(fH, fW) = face.shape[:2]# Get the face height and weight
			
			if fW < 20 or fH < 20: #Discard the small faces
				continue
			
			if opencv:#Using openCV to extract the frame embeddings
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
				emb.setInput(faceBlob)
				frameEmb = emb.forward()		
			else:#Using dlib to extract the embeddings
				rgb = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
				encodings = face_recognition.face_encodings(rgb,model="large")
				for enc in encodings:
					frameEmb=enc
			

			#Compare the face embedding of te frame with all faces registered on the dataset 
			matches = face_recognition.compare_faces(knownEmbeddings,frameEmb,tolerance=args["t"])
			
			#If a match has found
			if True in matches:
				counts={}
				indexes = [i for (i,b) in enumerate(matches)if b]
			#For each true in matches get the embedding
				
				for i in indexes:#For each match count the matches and get the name
					name = knownNames[i]
					counts[name] = counts.get(name,0) + 1
					
				name = max(counts,key=counts.get) # Get the name with more matches
				conf =(counts[name]*100)/samples[name]
				proba = "{}/{}".format(counts[name],samples[name])
				
				if conf/100 > args["p"]: 
					text = "{} : {}".format(name, proba)
				else:
					name="Unknown"
					
				matchesConfidences = {}
				for d in counts.items():
					matchesConfidences[d[0]] = "{}/{}".format(d[1],samples[d[0]])  
				
				if args["d"]:
					print("\nFrame#{}\nMatches = {}\n\nPredicted = {}\nConfidence = {}\n".format(count,matchesConfidences,name,proba))
					

			cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
			y = startY - 10 if startY - 10 > 10 else startY + 10	
			cv2.putText(frame, text, (startX, y),cv2.FONT_ITALIC,.45, (0, 00, 255), 2)
			
	text2="{} faces".format(noDetected)
	cv2.putText(frame,text2,(20,30),cv2.FONT_ITALIC,.60,(0,0,255),2)	
	cv2.imshow("Frame", frame)
	noDetected=0
	fps.update()
	
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
		

