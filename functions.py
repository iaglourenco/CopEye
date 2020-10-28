# All functions used by the program

import sys
import math
from datetime import datetime
import socket
import time
import os
import cv2
import traceback
from cv2 import data
import imutils
import numpy as np
import dlib
from imutils.face_utils.facealigner import FaceAligner
import face_recognition
import pickle
from multiprocessing import Process, Lock
import globalvar
import database
from database import Fugitivo,Artigo,Crime,Shot, UserDB




mutex = Lock()

MINIMAL_OCURRENCE = 5
TIMEOUT_2_SEND = 5

#Debug mode
DEBUG = True

IMAGE_TYPE_FRAME=1
IMAGE_TYPE_CROP=2
IMAGE_TYPE_DATASET_SAMPLE=3
INFO_TYPE=4

BUFFER_SIZE=1024

#IP and PORT to send the data
IP = "192.168.1.101"
DEFAULT_PORT=9700

#Filename of log
LOGNAME_ERR="logerr"
LOGNAME_INFO = "loginfo"
DETECTION_LOGNAME = "detectionLog"

#Clear old logs at the start
os.system("rm -f {}".format(LOGNAME_ERR))
if DEBUG:
	os.system("rm -f {}".format(LOGNAME_INFO))



defaultdb = database.CopEyeDatabase(r'./default_db.sqlite')
userdb = database.CopEyeDatabase(r'./user_db.sqlite')
defaultdb.init_database()
userdb.init_database()




def ex_info():
    #Get info about the exception
	exception = traceback.format_exc()
	if DEBUG:
		print(exception)
	write2Log(exception,LOGNAME_ERR)
	
def write2Log(text,logtype,print_terminal=False,supressDateHeader=False,append=True):
	"""
	Write data to the log
	- text: the text to write
	- logtype: wich logname to write
	- print_terminal: if True print text to the terminal
	- supressDateHeader: if True suppress the date header on the log
	- append: if True append text to the end of log file
	
	"""
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
	"""
	Calculate confidence based on the distance and tolerance
	- face_distance: the distance calculated
	- tolerance: the threashold of prediction 
	"""
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
	"""
	Send the data content through socket
	- dataType: The type of data to send (Text or File)
	"""
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
	"""
	Update detected with data provided by dataTuple
 	
	- detected: a empty dict, or a previously initiated dict
	- dataTuple: a tuple with these fields: (probability,frame,face_crop,faceComparedPath,frameNo)
	- return: a updated detected dictionary
	"""
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
	"""
	[THREAD LOOP]
	Wait for connetions from the app, and call the update function
	"""
	SEPARATOR ='/'
	s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	s.bind(('',5001)) # Accept connection from any address
	s.listen(1)
	print('Listening on',5001)
	while True:
		try:
			ns,address = s.accept()
			print('Connection from',address)
			received = ns.recv(BUFFER_SIZE)
			print(received)
			received = received.decode()
			filesize, crime, periculosity, name,age= received.split(SEPARATOR)
			filename='arquivo'

			try:
				os.mkdir('./datasets/{}'.format(name))
				write2Log('Creating folder for {}'.format(name),LOGNAME_INFO,True)

			except FileExistsError:
				write2Log('Appending photo to {} folder'.format(name),LOGNAME_INFO,True)
				pass
			with open('./datasets/{}/{}'.format(name,filename),'wb') as f:
				while True:
					recvBytes = ns.recv(int(filesize))
					if not recvBytes: break
					f.write(recvBytes)
			f.close()
			try:
				imgPath = ['./datasets/{}/{}'.format(name,filename)]
				update_user_encodings([name],imgPath)
				sqlite_add_fugitives(userdb,Fugitivo(name,age,periculosity),imgPath)
				globalvar.event.set()
	
			except Exception :
				print('Failed')
				ex_info()
				
			
		except KeyboardInterrupt:
			exit()
		except Exception as e:
			ex_info()
			pass

def __thread_call(detected,n):
	"""
	The thread call to send the data to the app in order, uses MUTEX for race condition protection
	"""

	#Mutex is used to ensure the order of send and to prevent to other thread send first
	# try to acquire the mutex
	with mutex:
		sendFrame(detected,n,INFO_TYPE)
		sendFrame(detected,n,IMAGE_TYPE_FRAME)
		sendFrame(detected,n,IMAGE_TYPE_CROP)
		sendFrame(detected,n,IMAGE_TYPE_DATASET_SAMPLE)

def updateFrequency(detected,history,timeouts):	
	"""
	Recebe o dict com as pessoas detectadas, historico atual e os timeouts verifica se esta na hora de enviar dados ou não
	retorna o historico e os timeouts atualizados
	"""
	for n in detected:
		history[n] = history.get(n,0)+1		
		timeouts[n]=timeouts.get(n,0)


		if DEBUG:
			write2Log("OCURRENCE OF {}={}\nACTUAL TIMEOUT= {}\n".format(n,history[n],time.process_time() - timeouts[n]),LOGNAME_INFO)

		if history[n] <= MINIMAL_OCURRENCE and time.process_time() - timeouts[n] > TIMEOUT_2_SEND:
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
	"""
	[DEPRECATED]
	Update the default pickle file with the extracted encoding, and the names
	- names: list of names in the same order as imagePaths
	- imagePaths: list of paths of face images
	"""
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
	print('Success')
	f.close()

def update_user_encodings(names,imagePaths):
	"""
	[DEPRECATED]
	Update the user pickle file with the extracted encoding, and the names
	- names: list of names in the same order as imagePaths
	- imagePaths: list of paths of face images
	"""
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
	f.close()

def sqlite_add_fugitives(db,fugitive: Fugitivo,imagePaths:list,artigo= -1):
	""" Add a fugitive to SQLite database
	 - ddb: SQLite database ( UserDB or DefaultDB)
	 - fugitive: Fugitive class
	 - imagePaths: list of images of fugitive
	 - artigo: ( Only for DefaultDB) Law article of the crime of the suspect 

	"""
	for imagePath in imagePaths:
		
		encoding = extract_embeddings_from_image_file(imagePath)
		if encoding is None:
			print('NULL encoding found',fugitive.nome)
			continue
		search = db.select('*','fugitivos','nome="{}" and idade="{}" and nivel_perigo="{}"'.format(fugitive.nome,fugitive.idade,fugitive.nivel_perigo))
		fugitive_id=0
		if len(search) != 0: # Possible duplicate in BD appending image and crime to first one
				fugitive_id = search[0][0]
		else:
			fugitive_id = db.insert_fugitivo(fugitive)
			
		db.insert_image(Shot(int(fugitive_id),imagePath,encoding))
		
		db.insert_crime(Crime(fugitive_id,artigo))

def extract_embeddings_from_image_file(imagePath: str):
	"""
	Extract the encoding of the image file in imagePath, 
	assuming that is one face in the frame
	- imagePath: the path to the image file
	"""

	#Load the face detector
	detector = cv2.dnn.readNetFromCaffe('models/face_detection_model/deploy.prototxt',
            'models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel')	
	
	image = cv2.imread(imagePath)
	if image is None:
		return image
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
	""" Extract the encoding of a np.ndarray photo,
		assuming that is one face in the frame
	- frame: a numpy.ndarray containing the photo
	"""
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

def align_faces(imagePaths: list):
	"""
	Align faces found in imagePaths, 
	overwrite original with the aligned face
    - imagePaths: list of paths of images to align

	"""
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

def kill_thread():
	"""Kills the receive thread"""
	if(__thread.is_alive()):
		__thread.kill()

def load_sqlite_db(db: database.CopEyeDatabase,data):
	"""Load the sqlite database and create the structure to use in detection"""
	dataset={}
	shots=[]
	crimes=[]


	#Load all articles from the database
	articles= db.select_all('articles')
	

	#Load all fugitives and create the structure
	db_fugitives = db.select_all('fugitives')
	for fugitive in db_fugitives:
		ident = fugitive[0]
		nome = fugitive[1]
		idade = fugitive[2]
		periculosidade = fugitive[3]
		
		#Select the images associated with the fugitive
		db_images = db.select('*','imagens','id={}'.format(ident))
		for image in db_images:
			uri = image[1]
			encoding = image[2]
			shots.append(database.Shot(ident, uri, encoding))	

		#Select the crimes associated with the fugitive
		db_crimes = db.select('*','crimes','id={}'.format(ident))
		for crime in db_crimes:
			artigo = crime[1]
			crimes.append(database.Crime(ident, artigo))

		#Add a item to the dict containing = Fugitive: ( listof(images) , listof(crimes) )
		dataset[database.Fugitivo(nome,idade,periculosidade,ident)]=(shots,crimes)


	#Return the dictionary and the list of articles
	return dataset,articles


__thread = Process(target=__receiveBytes)

def thread_listen():
	__thread.start()



