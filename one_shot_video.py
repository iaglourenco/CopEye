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
import math
from functions import *

ap = argparse.ArgumentParser()
ap.add_argument("-v",help="Path to the video",required=True)
ap.add_argument("-o",help="Path to the output video",required=True)
ap.add_argument("-d",help="Show information about processing",required=False,action="store_true")
ap.add_argument("-c",help="Minimum confidence to find face",required=False,type=float,default=0.8)
ap.add_argument("-p",help="Minimum confidence to predict a person, matches in dataset",required=False,type=float,default=0.85)
ap.add_argument("-t",help="tolerance of distance",required=False,type=float,default=0.6)
ap.add_argument("--interface",help="show minimal interface while running",required=False,action="store_true",default=False)
ap.add_argument("--interface2",help="show full interface while running",required=False,action="store_true",default=False)
ap.add_argument("--android",help="send data to the android app",required=False,action="store_true",default=False)
ap.add_argument("--log",help="save detections log to the disk, a echo to a file of the option '-d'",required=False,action="store_true",default=False)
args = vars(ap.parse_args())





if args["log"]:
	write2Log("#Frame no. - Best Match <-> Predicted = Distance : Probability - No. of matches",DETECTION_LOGNAME,supressDateHeader=True,append=False)


#Use dlib model
print("[INFO] - Using DLIB embedding model")
opencv = False
	
#Face detector, return the position of the faces in the image
print("[INFO] - Loading face detector")
detector = cv2.dnn.readNetFromCaffe("models/face_detection_model/deploy.prototxt",
			"models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")

sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(sp)

# Dataset with the embbedings knowned, frames will be compared with each face here

db_embeddings=[]
db_names=[]
db_facepaths=[]

user_embeddings=[]
user_names=[]
user_facepaths=[]

knownEmbeddings = []
knownNames = []
facePaths = []
noDetected=0

frameEmb = np.empty((128,))
proba = 0
timeout2Send=0

timeouts={}
history={}
detectedInFrame={}

print("[INFO] - Loading known embeddings")

#Loading to variables
sql_data,articles = load_sqlite_db(defaultdb)
for i in sql_data:
	f,imgs,crimes = sql_data.get(i,[])
	for i in imgs:
		db_embeddings.append(i.encoding)
		db_names.append(f.nome)
		db_facepaths.append(i.uri)


# db_data = pickle.loads(open("known/db_embeddings.pickle","rb").read()) 
# for e in db_data["embeddings"]:
# 	db_embeddings.append(e)

# for n in db_data["names"]:
# 	db_names.append(n)

# #facePaths for each person in the database
# for fp in db_data["facePaths"]:
# 	db_facepaths.append(fp)


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
frameNo=1
pause = False

if args['interface'] or args['interface2']:
	print("[INFO] - Starting video stream - Press 'p' to pause and 'q' to quit")
else:
	print("[INFO] - NO INTERFACE MODE - Press 'Ctrl+C' to quit")

if args["android"]:
	print("[INFO] - ANDROID MODE - Sending data to {}:{}\n".format(IP,DEFAULT_PORT))
	os.system("rm -f log/*")

globalvar.event.set()
try:
	while vc.isOpened():
		try:
			if globalvar.event.is_set():
				print('Updating user database')
				user_embeddings=[]
				user_names=[]
				user_facepaths=[]

				sql_data,articles = load_sqlite_db(userdb)
				for i in sql_data:
					f,imgs,crimes = sql_data.get(i,[])
					for i in imgs:
						user_embeddings.append(i.encoding)
						user_names.append(f.nome)
						user_facepaths.append(i.uri)

				# user_data = pickle.loads(open('known/user_embeddings.pickle','rb').read())
				# for e in user_data['embeddings']:
				# 	user_embeddings.append(e)
				# for n in user_data['names']:
				# 	user_names.append(n)
				# for fp in user_data['facePaths']:
				# 	user_facepaths.append(fp)

				knownEmbeddings = db_embeddings + user_embeddings
				knownNames = db_names + user_names
				facePaths = db_facepaths + user_facepaths
				globalvar.event.clear()


			ret,frame = vc.read()# Read a frame
			frameNo+=1
			print("{}/{}".format(frameNo,vc.get(cv2.CAP_PROP_FRAME_COUNT)),end='\r')
			if ret == True and frameNo % 2 == 0:
				
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

						# al = np.copy(frame)
						# gray=cv2.cvtColor(al,cv2.COLOR_BGR2GRAY)
						
						# face = fa.align(al,
						# gray,
						# dlib.rectangle(startX,startY,endX,endY))		
							
						if noDetected > 0 and args['interface2']:
							cv2.imshow("Face#{}".format(f),face)
						
						frameEmb = np.empty(128,)
								
						#Using dlib to extract the embeddings
						rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
						encodings=[]
						# locations = face_recognition.face_locations(rgb,model="cnn")
						# encodings = face_recognition.face_encodings(rgb,num_jitters=1,model="large")
						encodings = face_recognition.face_encodings(rgb,[(startY,endX,endY,startX)],num_jitters=2,model="large")							
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
								distance-=distance/2
								ind = matchInd
								name = knownNames[ind]
								probability = distance2conf(distance,args["t"])
						else:
							distance+=distance/3
								
								
						name = knownNames[ind]				
						faceComparedPath = facePaths[ind]
						
						if probability >= args["p"] : 
							text = "#{}-{} : {:.2f}%".format(f,name, probability*100)
							if args['android']:
								detectedInFrame = createDetectedStruct(detectedInFrame,(probability,name,frameOut,face,faceComparedPath,frameNo))
						else:
							name="Unknown"
						
						y = startY - 10 if startY - 10 > 10 else startY + 10
						cv2.rectangle(frameOut, (startX, startY), (endX, endY),(0, 0, 255), 2)
						cv2.putText(frameOut, text, (startX, y),cv2.FONT_ITALIC,.45, (0, 255, 255), 2)

						if len(detectedInFrame) > 0 and time.process_time() - timeout2Send > 2 and args["android"]:
							print('Checking...')
							timeout2Send=time.process_time()
							history,timeouts = updateFrequency(detectedInFrame,history,timeouts)
							detectedInFrame.clear()
							
							
						if args['interface2']:
							faceCompared = cv2.imread(faceComparedPath)
							if not faceCompared is None:
								imutils.resize(faceCompared,width=600,height=600)
								cv2.imshow("Face#{} Best match".format(f),faceCompared)
						

						if args["d"]:
								print("\nFace#{}\nLooks like = {}\nPredicted = {}\nDistance = {}\nProbability = {:.2f}%\nMatch count = {}\n".format(f,knownNames[ind],name,distance,probability*100,matchCount.get(name,"NULL")))

						if args['log']:
							detectionLog = "#{} - {} <-> {} = {} : {:.2f}% - {} match(s)".format(frameNo,knownNames[ind],name,distance,probability*100,matchCount.get(name,"NULL"))
							write2Log(detectionLog,DETECTION_LOGNAME,supressDateHeader=True)

				noDetected=0
				out.write(frameOut)
				if args['interface2'] or args['interface']:
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
					print("\nStopped by the user")
					raise KeyboardInterrupt
			elif ret == False:
				raise KeyboardInterrupt
		except Exception:
			ex_info()
except KeyboardInterrupt:
	fps.stop()
	print("[INFO] - elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] - approx. FPS: {:.2f}".format(fps.fps()))
	vc.release()
	out.release()
	cv2.destroyAllWindows()

	time.sleep(2)
	exit()


	 