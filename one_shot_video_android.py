import socket
import os
import time
from datetime import datetime
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
import sys
import face_recognition
import dlib
import argparse
import math
from functions import *

BUFFER_SIZE = 1024 # send 4096 bytes each time step
host = "192.168.1.101"
port = 9700
timeoutEachPerson = 0
frames = []
history={}
N_OCURRENCE=10
frequency=[]


def sendFrame(p,name2send):

	toSend = p.get(name2send)
	
	now = datetime.now()

	tempo = now.strftime("%d-%m-%Y %H:%M:%S")
	path1 = "./log/{}-{}-{}.jpg".format(count, name2send, tempo)
	path2 = "./log/{}-{}-{}_face_crop.jpg".format(count, name2send, tempo)
	msg = " " + name2send + "\n" + " "+ tempo + "\n" +" "+str(round(toSend[0]*100,2))+"%" + "\n"+ "\0"
	msg = msg.ljust(1024,"0")
	try:
		cv2.imwrite(path1, toSend[1])
		cv2.imwrite(path2,toSend[2])
	except Exception as e:
		print("[ERROR] - Failed to save log")
		ex_info()
		
	enviaBytes(msg, "1")
	enviaBytes(path1, "2")
	enviaBytes(path2, "2")
	enviaBytes(faceComparedPath,"2")
	
		


def enviaBytes(sendBytes, opt):


		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)		
		s.connect((host, port))	

		if opt == "1":
			s.send(sendBytes.encode())

		else:
			filesize = os.path.getsize(sendBytes)
			with open(sendBytes, "rb") as f:
				while(True):
					# read the bytes from the file
					bytes_read = f.read(BUFFER_SIZE)
					if not bytes_read:
						break
					s.sendall(bytes_read)
					
		s.close()		
	



def checkDetectionFrequency(p):	
	global timeoutEachPerson
	for n in p:
		history[n] = history.get(n,0)+1

		ocurrence = history.get(n,0)
		global timeoutEachPerson
		if ocurrence <= N_OCURRENCE and time.time() - timeoutEachPerson > 5:
			sendFrame(p,n)
			timeoutEachPerson = time.time()
			frames.clear()

ap = argparse.ArgumentParser()
ap.add_argument("-v",help="Path to the video",required=True)
ap.add_argument("-o",help="Path to the output video",required=True)
ap.add_argument("-d",help="Show information about processing",required=False,action="store_true")
ap.add_argument("-c",help="Minimum confidence to find face",required=False,type=float,default=0.8)
ap.add_argument("-p",help="Minimum confidence to predict a person, matches in dataset",required=False,type=float,default=0.85)
ap.add_argument("-t",help="tolerance of distance",required=False,type=float,default=0.6)
ap.add_argument("--interface",help="show interface while running",required=False,action="store_true",default=False)
ap.add_argument("--opencv",help="Use opencv model to extract embeddings",required=False,action="store_true")
args = vars(ap.parse_args())


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
timeout2Send=0
detectedPersons ={}

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
count=1
pause = False

if args['interface']:
	print("[INFO] - Starting video stream - Press 'p' to pause and 'q' to quit")
else:
	print("[INFO] - NO INTERFACE MODE - Press 'Ctrl+C' to quit")

try:
	while vc.isOpened():

		ret,frame = vc.read()
		count+=1
		print("{}/{}".format(count,vc.get(cv2.CAP_PROP_FRAME_COUNT)),end='\r')
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
					
					# face = frame[startY:endY, startX:endX]
					
					# (fH, fW) = face.shape[:2]# Get the face height and weight			
					# if fW > 250 or fH > 340 or fW < 20 or fH < 20:
					# 	continue

					al = np.copy(frame)
					gray=cv2.cvtColor(al,cv2.COLOR_BGR2GRAY)
					
					face = fa.align(al,
					gray,
					dlib.rectangle(startX,startY,endX,endY))		
						
					#if noDetected > 0 and args['interface']:
					#	cv2.imshow("Face#{}".format(f),face)
					
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
						encodings = face_recognition.face_encodings(rgb,num_jitters=1,model="large")
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
								matchInfo[n+"index"] = i
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
					else:
						distance+=distance
							
							
					name = knownNames[ind]				
					faceComparedPath = facePaths[ind]
					
					if probability >= args["p"] : 
						text = "#{}-{} : {:.2f}%".format(f,name, probability*100)
						frequency.append((probability,name,frameOut,face))
					else:
						name="Unknown"

					if len(frequency) > 0 and time.time() - timeout2Send > 2 and noDetected>0:
						# shot[0]=probability
						# shot[1]=name
						# shot[2]=frameOut
						# shot[3]=faceCrop
						timeout2Send=time.time()
						for shot in frequency :
							p = detectedPersons.get(shot[1])
							if p == None:
								detectedPersons[shot[1]]=(shot[0],shot[2],shot[3])
							elif p[0] < shot[0]:
								detectedPersons[shot[1]]=(shot[0],shot[2],shot[3])
						checkDetectionFrequency(detectedPersons)
						frequency.clear()
						detectedPersons.clear()


					if args['interface']:
						faceCompared = cv2.imread(faceComparedPath)
						#if not faceCompared is None:
						#	imutils.resize(faceCompared,width=600,height=600)
						#	cv2.imshow("Face#{} Best match".format(f),faceCompared)
					


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
	time.sleep(2)
	exit()
# except Exception as e:
# 	print("[ERROR] - Error during execution")
# 	ex_info()


