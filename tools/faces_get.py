
from imutils.video import VideoStream 
import imutils
import cv2
import time
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import paths
import face_recognition
import numpy as np
import argparse
import os
import dlib
import pickle
from progressbar import ProgressBar,ETA,FileTransferSpeed

# Get faces from webcam and align them
# I'm assuming that have only one face in the frame
#
# @author: Iago Lourenço - @iaglourenco


ap = argparse.ArgumentParser()
ap.add_argument("-d","--dest",required=True,help="path to the folder of the person")
ap.add_argument("-a","--amount",help="amount of images to take",default=100,type=int)
ap.add_argument("-t","--time",help="delay between shots",type=float,default=0.5)
args = vars(ap.parse_args())

detectorAlign = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../one_shot/models/shape_predictor_5_face_landmarks.dat")
fa = FaceAligner(predictor,desiredFaceHeight=256)

detectorFace = cv2.dnn.readNetFromCaffe("../one_shot/models/face_detection_model/deploy.prototxt","../one_shot/models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
vs = VideoStream(src=0,resolution=(1920,1080)).start()
time.sleep(2)
total=1
images = list()

def align(shot,tot):
    gray = cv2.cvtColor(shot,cv2.COLOR_BGR2GRAY)
    rects = detectorAlign(gray,2)
    for rect in rects:
        (x,y,w,h) = rect_to_bb(rect)
        aligned = fa.align(shot,gray,rect)
        return aligned
  
    return shot

try:
    f = open(args["dest"]+"/dataset.nfo","rb")
    dataSize = pickle.load(f)
    print("Detected {} images on this dataset".format(dataSize))
except FileNotFoundError :
    f = open(args["dest"]+"/dataset.nfo","xb")
    dataSize=1
    pickle.dump(dataSize,f)
except EOFError:
    dataSize=0

f.close()
f = open(args["dest"]+"/dataset.nfo","wb")

bar = ProgressBar(args["amount"]).start()
bar.widgets.append(ETA())
bar.widgets.append(FileTransferSpeed(unit="frames"))
print("Detecting faces")
while total < args["amount"]:
    frame = vs.read()
    frame = imutils.resize(frame,width=300,height=300)
    cv2.imshow("Frame",frame)
    noDetected=0
    
    imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
    detectorFace.setInput(imageBlob)
    detections = detectorFace.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.9:
            noDetected+=1
        if noDetected>1:
            break

    i = np.argmax(detections[0, 0, :, 2])
    confidence = detections[0, 0, i, 2]
    
    if noDetected == 1 and  confidence > 0.9:
        bar.update(total+1)
        total+=1
        images.append(frame)
    else:
        print("[Stopped]".ljust(bar.term_width,' '),end='\r')
        bar.update()
            
     
    time.sleep(args["time"])
    
    key=cv2.waitKey(2) & 0xFF
    if key == ord("q") or total >= args["amount"]:
        bar.finish()
        pbar = ProgressBar(maxval=len(images)).start()
        pbar.widgets.append(ETA())
        pbar.widgets.append(FileTransferSpeed(unit="photos"))
        vs.stop()
        cv2.destroyAllWindows()
        print("Aligning faces")
        total=0
        for im in images:
            face=align(im,dataSize)
            if im.shape != face.shape:
                cv2.imwrite(args["dest"]+"/aligned{}.jpg".format(str(dataSize+1).zfill(3)),face)
                pbar.update(total+1)
            dataSize+=1
            total+=1
        break

pickle.dump(dataSize,f)
f.close()
pbar.finish()

    



         