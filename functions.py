import sys
import math
from datetime import datetime
import socket
import time
import os
import cv2
import traceback
import imutils
import numpy as np
import dlib
from imutils.face_utils.facealigner import FaceAligner
import face_recognition
import pickle
from multiprocessing import Process, Lock


mutex = Lock()

DATABASE_IS_UPDATED = False

MINIMAL_OCURRENCE = 5
TIMEOUT_2_SEND = 5

#Debug mode
DEBUG = False

IMAGE_TYPE_FRAME=1
IMAGE_TYPE_CROP=2
IMAGE_TYPE_DATASET_SAMPLE=3
INFO_TYPE=4

BUFFER_SIZE=1024

#IP and PORT to send the data
IP = "192.168.1.101"
DEFAULT_PORT=9700


#Filenames of log
LOGNAME_ERR="logerr"
LOGNAME_INFO = "loginfo"
DETECTION_LOGNAME = "detectionLog"

#Clear old logs at the start
os.system("rm -f {}".format(LOGNAME_ERR))
if DEBUG:
	os.system("rm -f {}".format(LOGNAME_INFO))


def ex_info():
    #Get info about the exception
	exception = traceback.format_exc()
	print(exception)
	write2Log(exception,LOGNAME_ERR)
	
def write2Log(text,logtype,print_terminal=False,supressDateHeader=False,append=True):
	#write data to the log
	header="\r"
	if supressDateHeader == False:
		header = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

	if append:
		os.system("echo '{}\n{}' >> {}".format(header,text,logtype))
	else:
		os.system("echo '{}\n{}' > {}".format(header,text,logtype))

	if(print_terminal):
		print(text)

def distance2conf(face_distance,tolerance):
    # Calculate confidence based on the distance and tolerance
	if face_distance > tolerance:
		range = (1.0 - tolerance)
		linear_val = (1.0 - face_distance) / (range * 2.0)
		return linear_val
	else:
		range = tolerance
		linear_val = 1.0 - (face_distance / (range * 2.0))
		return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) *2,0.2))

def sendFrame(detected,name,typeOfSend):
	#Convert and prepare and send data through socket

	toSend = detected.get(name)
	
	probability = toSend[0]
	frame = toSend[1]
	face_crop = toSend[2]
	faceComparedPath = toSend[3]
	frameN = toSend[4]

	now = datetime.now()
	tempo = now.strftime("%d-%m-%Y %H:%M:%S")
	

	if typeOfSend == IMAGE_TYPE_FRAME:
		pathFrame = "./log/{}-{}-{}.jpg".format(frameN, name, tempo)
		try:
			cv2.imwrite(pathFrame, frame)
			
		except IOError:
			print("[ERROR] - Failed to save frame to log")
			ex_info()
			raise IOError
		__sendBytes(pathFrame, typeOfSend)
			
	elif typeOfSend == IMAGE_TYPE_CROP:
		pathCrop = "./log/{}-{}-{}_face_crop.jpg".format(frameN, name, tempo)
		try:
			cv2.imwrite(pathCrop, face_crop)
			
		except IOError:
			print("[ERROR] - Failed to save face crop to log")
			ex_info()
			raise IOError
		__sendBytes(pathCrop, typeOfSend)
	
	elif typeOfSend == IMAGE_TYPE_DATASET_SAMPLE:
		__sendBytes(faceComparedPath,typeOfSend)
	
	elif typeOfSend == INFO_TYPE:
		msg = " " + name + "\n" + " "+ tempo + "\n" +" "+str(round(probability*100,2))+"%" + "\n"+ "\0"
		msg = msg.ljust(1024,"0")
		__sendBytes(msg, typeOfSend)

	else:
		raise ValueError

def __sendBytes(data, dataType):
	#Send the content through socket
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)		
		s.connect((IP, DEFAULT_PORT))	

		if dataType == INFO_TYPE:
			s.send(data.encode())

		else:
			filesize = os.path.getsize(data)
			with open(data, "rb") as f:
				while(True):
					# read the bytes from the file
					bytes_read = f.read(BUFFER_SIZE)
					if not bytes_read:
						break
					s.sendall(bytes_read)
					
	
		s.close()	

def createDetectedStruct(detected,dataTuple):
	# Update detected with data provided by dataTuple
 	# :param detected: a empty dict, or a previously initiated dict
	# :param dataTuple: a tuple with these fields: (probability,frame,face_crop,faceComparedPath,frameNo)
	# :return: a updated detected dictionary
	probability =dataTuple[0]
	name=dataTuple[1]
	frame=dataTuple[2]
	face_crop=dataTuple[3]
	faceComparedPath=dataTuple[4]
	frameNo=dataTuple[5]

	p = detected.get(name)
	if p == None:
		detected[name]=(probability,frame,face_crop,faceComparedPath,frameNo)
	elif p[0] < probability:
		detected[name]=(probability,frame,face_crop,faceComparedPath,frameNo)

	return detected

def __receiveBytes():
	SEPARATOR ='/'
	s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	print('Listening')
	s.bind(('192.168.1.113',5001))
	s.listen(1)
	while True:
		try:
			ns,address = s.accept()
			print('Connection from',address)
			received = ns.recv(BUFFER_SIZE)
			print(received)
			received = received.decode()
			filesize, crime, skin_tone, periculosity, name = received.split(SEPARATOR)
			filename='arquivo'

			try:
				os.mkdir('./datasets/{}'.format(name))
			except Exception:
				pass
			with open('./datasets/{}/{}'.format(name,filename),'wb') as f:
				while True:
					recvBytes = ns.recv(BUFFER_SIZE)
					if not recvBytes: break
					f.write(recvBytes)
			f.close()
			update_user_encodings([name],['./datasets/{}/{}'.format(name,filename)])
			DATABASE_IS_UPDATED = False
		except KeyboardInterrupt:
			exit()
		except Exception as e:
			ex_info()
			pass

def __thread_call(detected,n):
	

	#Mutex is used to ensure the order of send and to prevent to other thread send first
	# try to acquire the mutex
	with mutex:
		sendFrame(detected,n,INFO_TYPE)
		sendFrame(detected,n,IMAGE_TYPE_FRAME)
		sendFrame(detected,n,IMAGE_TYPE_CROP)
		sendFrame(detected,n,IMAGE_TYPE_DATASET_SAMPLE)

def updateFrequency(detected,history,timeouts):	
	#Recebe o dict com as pessoas detectadas, historico atual e os timeouts verifica se esta na hora de enviar dados ou não
	#retorna o historico e os timeouts atualizados
	for n in detected:
		history[n] = history.get(n,0)+1
		timeouts[n]=timeouts.get(n,time.process_time())


		if DEBUG:
			write2Log("OCURRENCE OF {}={}\nACTUAL TIMEOUT= {}\n".format(n,history[n],timeouts[n]),LOGNAME_INFO)


		if history[n] == MINIMAL_OCURRENCE and time.process_time() - timeouts[n] > TIMEOUT_2_SEND:
			history[n]=0
			timeouts[n]=time.process_time()	
			
		elif history[n] == 2:
			timeouts[n]=time.process_time()
			if DEBUG:
				write2Log("Trying to send to {}:{}\n".format(IP,DEFAULT_PORT),LOGNAME_INFO,True)
			try:
				
				
				# Start a thread to send the data
				t=Process(target=__thread_call,args=(detected,n))
				t.start()
				
				# sendFrame(detected,n,INFO_TYPE)
				# sendFrame(detected,n,IMAGE_TYPE_FRAME)
				# sendFrame(detected,n,IMAGE_TYPE_CROP)
				# sendFrame(detected,n,IMAGE_TYPE_DATASET_SAMPLE)
				if DEBUG:
					write2Log("Data sended to {}:{}\n".format(IP,DEFAULT_PORT),LOGNAME_INFO,True)

			except ConnectionRefusedError:
				print("[ERROR] - Failed to connect to the app")
				ex_info()
			except OSError:
				print("[ERROR] - Failed to connect to the app")
				ex_info()	

			
	return history,timeouts
			
def update_db_encodings(names,imagePaths):
	#update the default database of faces
	try:
		knownEmbeddings = []
		knownNames=[]
		facePaths=[]
		f = open('known/db_embeddings.pickle','rb')
		db_enc = pickle.loads(f.read())
		f.close()
		print('Reading and updating file')
		for e in db_enc["embeddings"]:
			knownEmbeddings.append(e)

		for n in db_enc["names"]:
			knownNames.append(n)

		for fp in db_enc["facePaths"]:
			facePaths.append(fp)
	except FileNotFoundError:
		print('Creating file')
	
	
	for (i,facePath) in enumerate(imagePaths):
		enc = extract_embeddings_from_image_file(facePath)
		if enc is not None:
			knownEmbeddings.append(enc)
			knownNames.append(names[i])
			facePaths.append(facePath)
		
	f = open('known/db_embeddings.pickle','wb')
	data = {'embeddings':knownEmbeddings,
			'names':knownNames,
			'facePaths':facePaths}
	
	f.write(pickle.dumps(data))
	print('Sucess')
	f.close()

def update_user_encodings(names,imagePaths):
	#update the user database of faces
	try:
		knownEmbeddings = []
		knownNames=[]
		facePaths=[]
		f = open('known/user_embeddings.pickle','rb')
		user_enc = pickle.loads(f.read())
		f.close()
		print('Reading and updating file')
		for e in user_enc["embeddings"]:
			knownEmbeddings.append(e)

		for n in user_enc["names"]:
			knownNames.append(n)

		for fp in user_enc["facePaths"]:
			facePaths.append(fp)
	except FileNotFoundError:
		print('Creating file')
	
	
	for (i,facePath) in enumerate(imagePaths):
		enc = extract_embeddings_from_image_file(facePath)
		if enc is not None:
			knownEmbeddings.append(enc)
			knownNames.append(names[i])
			facePaths.append(facePath)
	
	f = open('known/user_embeddings.pickle','wb')
	data = {'embeddings':knownEmbeddings,
			'names':knownNames,
			'facePaths':facePaths}
	f.write(pickle.dumps(data))
	print('Sucess')
	f.close()

def extract_embeddings_from_image_file(imagePath):
	#Return the encoding for a imgae file, assume that is one face in the frame

	#Load the face detector
	detector = cv2.dnn.readNetFromCaffe('models/face_detection_model/deploy.prototxt',
            'models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel')	
	
	image = cv2.imread(imagePath)
	
	#Resize the image to put in the detector
	image = imutils.resize(image,width=600)
	(h,w) = image.shape[:2]
	imgBlob = cv2.dnn.blobFromImage(
        cv2.resize(image,(300,300)),
        1.0,
        (300,300),
        (104.0,177.0,123.0),
        swapRB=False,
        crop=False
        )
	
	#Perform detection
	detector.setInput(imgBlob)
	detections = detector.forward()
	
	#Assuming that is just one face on the image
	if len(detections)>0:
		j = np.argmax(detections[0,0,:,2])
		confidence = detections[0,0,j,2]
		if confidence > 0.6:
			box = detections[0,0,j,3:7] * np.array([w,h,w,h])
			(startX,startY,endX,endY) = box.astype('int')
			
			(fH,fW) = image[startY:endY,startX:endX].shape[:2]
			if fW < 20 or fH <20:
				return None

			#extract the encoding
			face_encodings = __extract_encoding(image)
		
			return face_encodings
	else:
		return None

def __extract_encoding(frame):

	#Return a encoding for the frame, assuming that is one face in the frame

	rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	encodings=[]
	locations = face_recognition.face_locations(rgb,model="hog")
	if len(locations) == 0:
		return None
	# encodings = face_recognition.face_encodings(rgb,[(startY,endX,endY,startX)],num_jitters=2,model="large")
	encodings = face_recognition.face_encodings(rgb,locations,num_jitters=10,model="large")
	for enc in encodings:
		emb=enc

	return emb

def align_faces(imagePaths):
	#Align faces found in imagePaths, overwrite original with the aligned face

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
	fa = FaceAligner(predictor,desiredFaceHeight=256)

	for i,imagePath in enumerate(imagePaths):
	    print("[INFO] - Aligning face #{}".format(i))
	    image = cv2.imread(imagePath)
	    image = imutils.resize(image,width=800)
	    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	    rects = detector(gray,2)
	    for rect in rects:
	        image = fa.align(image,gray,rect)
	        cv2.imwrite(imagePath,image) 


Process(target=__receiveBytes).start()
