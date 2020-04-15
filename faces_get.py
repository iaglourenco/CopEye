
from imutils.video import VideoStream 
import imutils
import cv2
import time
# Get faces from webcam
# I'm assuming that have only one face in the frame
#
# @author: Iago Lourenço - @iaglourenco
import face_recognition
import numpy as np
import argparse
import os
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dest",required=True,help="path to the folder of the person")
ap.add_argument("-a","--amount",help="amount of images to take",default=100,type=int)
ap.add_argument("-t","--time",help="delay between shots",type=float,default=0.5)
args = vars(ap.parse_args())

detector = cv2.dnn.readNetFromCaffe("models/face_detection_model/deploy.prototxt","models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
vs = VideoStream(src=0).start()
time.sleep(2)
total=0

while total < args["amount"]:
    os.system("clear")
    print("Searching for faces...")
    frame = vs.read()
    frame = imutils.resize(frame,width=300,height=300)
    cv2.imshow("Frame",frame)
    
    imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()
    i = np.argmax(detections[0, 0, :, 2])
    confidence = detections[0, 0, i, 2]

    if len(detections) > 0 and  confidence > 0.8    :
        print("Face found! - {}/{} pictures taken!".format(total+1,args["amount"]))
        cv2.imwrite(args["dest"]+"{}.jpg".format(str(total).zfill(5)),frame)
        total+=1
    else:
        print("No faces found")
    time.sleep(args["time"])
    
    key=cv2.waitKey(2) & 0xFF
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()
vs.stop()
