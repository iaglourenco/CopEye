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
import math


ap = argparse.ArgumentParser()
ap.add_argument("-v",help="Path to the video",required=True)
ap.add_argument("-o",help="Path to the output video",required=True)
ap.add_argument("-d",help="Show information about processing",required=False,action="store_true")
ap.add_argument("-c",help="Minimum confidence to find face",required=False,type=float,default=0.8)
ap.add_argument("-p",help="Minimum confidence to predict a person, matches in dataset",required=False,type=float,default=0.55)
ap.add_argument("-t",help="tolerance of distance",required=False,type=float,default=0.6)
ap.add_argument("--interface",help="show interface while running",required=False,action="store_true",default=False)
ap.add_argument("--opencv",help="Use opencv model to extract embeddings",required=False,action="store_true")
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
	#Use openCV model
    opencv=True
    print("[INFO] - Loading OpenCV embedding model")
    emb = cv2.dnn.readNetFromTorch("models/nn4.small1.v1.t7")
else:
	#Use dlib model
    print("[INFO] - Using DLIB embedding model")
    opencv = False
	
print("[INFO] - Loading face detector")
detector = cv2.dnn.readNetFromCaffe("models/face_detection_model/deploy.prototxt",
			"models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")

sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(sp)


print("[INFO] - Loading known embeddings")
data = pickle.loads(open("known/embeddings.pickle","rb").read()) 
knownEmbeddings = []
knownNames = []
facePaths = [] 
noDetected=0
frameEmb = np.empty((128,))
proba = 0

for e in data["embeddings"]:
	knownEmbeddings.append(e)

for n in data["names"]:
	knownNames.append(n)

for fp in data["facePaths"]:
	facePaths.append(fp)


print("[INFO] - Starting video read")
vc = VideoCapture(args["v"])
if vc.isOpened() == False:
	print("[ERROR] - Failed to open video")
	exit()

ret,frame = vc.read()
frame = imutils.resize(frame, width=600)

#If the video is rotated 
#frame = imutils.rotate(frame,angle=90)                

out = VideoWriter(args["o"]+".avi",cv2.VideoWriter_fourcc(*'XVID'),vc.get(cv2.CAP_PROP_FPS)/2,(frame.shape[1],frame.shape[0]))

fps = FPS().start()
bar = ProgressBar(vc.get(cv2.CAP_PROP_FRAME_COUNT)+2).start()
bar.widgets.append(ETA())
count=1
pause = False

if args['interface']:
	print("[INFO] - Starting video stream - Press 'p' to pause and 'q' to quit")
else:
	print("[INFO] - NO INTERFACE MODE - Press 'Ctrl+C' to quit")

try:
	while vc.isOpened():

		ret,frame = vc.read()
		bar.update(count)
		count+=1
		if ret == True and count % 2 == 0:
			
			frame = imutils.resize(frame, width=600)
			#frame = imutils.rotate(frame,angle=90)
			(h, w) = frame.shape[:2]
			frameOut = np.copy(frame)
			
			imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
			detector.setInput(imageBlob)
			detections = detector.forward()
			
			for f in range(0, detections.shape[2]):
				name="Unknown"
				text = "{}".format(name)
				confidence = detections[0, 0, f, 2]
				if confidence > args["c"]:
					noDetected+=1
					box = detections[0, 0, f, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					face = frame[startY:endY, startX:endX]
					(fH, fW) = face.shape[:2]# Get the face height and weight			
					if fW > 250 or fH > 340 or fW < 20 or fH < 20:
						continue

					al = np.copy(frame)
					gray=cv2.cvtColor(al,cv2.COLOR_BGR2GRAY)
					
					face = fa.align(al,
					gray,
					dlib.rectangle(startX,startY,endX,endY))		
						
					if noDetected > 0 and args['interface']:
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
						locations = face_recognition.face_locations(rgb,model="cnn")
						encodings = face_recognition.face_encodings(rgb,num_jitters=2,model="large")
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
					probability = distance2conf(distance,args["t"])

					if len(matchCount) > 0:
						matchName = max(matchCount,key=matchCount.get)
						matchInd = matchInfo.get(matchName+"index")
						matchDis = matchInfo.get(matchName+"distance")
						nOfMatches = (matchCount.get(matchName))
						if matchDis <= distance and nOfMatches > 1:
							ind = matchInd
							name = knownNames[ind]
							probability = distance2conf(distance,args["t"])
							
							
							
					name = knownNames[ind]				
					faceComparedPath = facePaths[ind]
					
					if probability >= args["p"] : 
						text = "#{}-{} : {:.2f}%".format(f,name, probability*100)
					else:
						name="Unknown"


					if args['interface']:
						faceCompared = cv2.imread(faceComparedPath)
						if not faceCompared is None:
							imutils.resize(faceCompared,width=600,height=600)
							cv2.imshow("Face#{} Best match".format(f),faceCompared)
					


					if args["d"]:
							print("\nFace#{}\nLooks like = {}\nPredicted = {}\nDistance = {}\nProbability = {:.2f}%\nMatch count = {}\n".format(f,knownNames[ind],name,distance,probability*100,matchCount.get(name,-1)))

					y = startY - 10 if startY - 10 > 10 else startY + 10
					cv2.rectangle(frameOut, (startX, startY), (endX, endY),(0, 0, 255), 2)
					cv2.putText(frameOut, text, (startX, y),cv2.FONT_ITALIC,.45, (0, 255, 255), 2)
			
			noDetected=0
			out.write(frameOut)
			if args['interface']:
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
				raise KeyboardInterrupt

except KeyboardInterrupt:
	print("\nStopped by the user")
	fps.stop()
	print("[INFO] - elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] - approx. FPS: {:.2f}".format(fps.fps()))
	vc.release()
	out.release()
	cv2.destroyAllWindows()
	bar.finish()
	time.sleep(2)
	exit()



	 