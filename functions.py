import sys
import math
from datetime import datetime
import socket
import time
import os
import cv2
import traceback

MINIMAL_OCURRENCE = 5
TIMEOUT_2_SEND = 5

DEBUG = False

IMAGE_TYPE_FRAME=1
IMAGE_TYPE_CROP=2
IMAGE_TYPE_DATASET_SAMPLE=3
INFO_TYPE=4

BUFFER_SIZE=1024

IP = "192.168.1.101"
DEFAULT_PORT=9700

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
		_sendBytes(pathFrame, typeOfSend)
			
	elif typeOfSend == IMAGE_TYPE_CROP:
		pathCrop = "./log/{}-{}-{}_face_crop.jpg".format(frameN, name, tempo)
		try:
			cv2.imwrite(pathCrop, face_crop)
			
		except IOError:
			print("[ERROR] - Failed to save face crop to log")
			ex_info()
			raise IOError
		_sendBytes(pathCrop, typeOfSend)
	
	elif typeOfSend == IMAGE_TYPE_DATASET_SAMPLE:
		_sendBytes(faceComparedPath,typeOfSend)
	
	elif typeOfSend == INFO_TYPE:
		msg = " " + name + "\n" + " "+ tempo + "\n" +" "+str(round(probability*100,2))+"%" + "\n"+ "\0"
		msg = msg.ljust(1024,"0")
		_sendBytes(msg, typeOfSend)

	else:
		raise ValueError



def _sendBytes(data, dataType):
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


def updateFrequency(detected,history,timeouts):	
	#Recebe o dict com as pessoas detectadas, historico atual e os timeouts verifica se esta na hora de enviar dados ou não
	#retorna o historico e os timeouts atualizados
	for n in detected:
		history[n] = history.get(n,0)+1
		timeouts[n]=timeouts.get(n,time.clock())


		if DEBUG:
			write2Log("OCURRENCE OF {}={}\nACTUAL TIMEOUT= {}\n".format(n,history[n],timeouts[n]),LOGNAME_INFO)


		if history[n] == MINIMAL_OCURRENCE and time.clock() - timeouts[n] > TIMEOUT_2_SEND:
			history[n]=0
			timeouts[n]=time.clock()	
			
		elif history[n] == 1:
			timeouts[n]=time.clock()
			if DEBUG:
				write2Log("Trying to send to {}:{}\n".format(IP,DEFAULT_PORT),LOGNAME_INFO,True)
			try:
				
				sendFrame(detected,n,INFO_TYPE)
				sendFrame(detected,n,IMAGE_TYPE_FRAME)
				sendFrame(detected,n,IMAGE_TYPE_CROP)
				sendFrame(detected,n,IMAGE_TYPE_DATASET_SAMPLE)
				if DEBUG:
					write2Log("Data sended to {}:{}\n".format(IP,DEFAULT_PORT),LOGNAME_INFO,True)

			except ConnectionRefusedError:
				print("[ERROR] - Failed to connect to the app")
				ex_info()
			except OSError:
				print("[ERROR] - Failed to connect to the app")
				ex_info()	

			
	return history,timeouts
			