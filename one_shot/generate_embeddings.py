from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os
from progressbar import ProgressBar,ETA,FileTransferSpeed
import time
import face_recognition
import dlib
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("--dlib",help="Use dlib's to extract embeddings",required=False,action="store_true")
ap.add_argument("--data","-d",help="Path to the dataset",required=True)
args = vars(ap.parse_args())


#Face detector get the face locations on a frame image
print("[INFO] - Loading face detector")
detector = cv2.dnn.readNetFromCaffe("models/face_detection_model/deploy.prototxt",
            "models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")


## Get the images path
print("[INFO] - Loading faces")
imagePaths = list(paths.list_images(args.get("data")))
knownEmbeddings = []
knownNames = []
samples = {}

if args["dlib"]:
    #Use dlib embeddings extractor
    print("[INFO] - Using DLIB embedding model")
    opencv = False
else:
    #Use OpenCV embeddings extractor
    opencv=True
    print("[INFO] - Loading OpenCV embedding model")
    emb = cv2.dnn.readNetFromTorch("models/nn4.small1.v1.t7")

#Progress monitor
total = 0
bar = ProgressBar(maxval=len(imagePaths)).start()
bar.widgets.append(ETA())
bar.widgets.append(FileTransferSpeed(unit="img"))

#For each image path load the image and extract the face embedding
for (i,imagePath) in enumerate(imagePaths):   
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    #Resize for the detector
    image = imutils.resize(image,width=600)
    (h,w)= image.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image,(300,300)),
        1.0,
        (300,300),
        (104.0,177.0,123.0),
        swapRB=False,
        crop=False
        )

    #Detect face locations
    detector.setInput(imageBlob)
    detections = detector.forward()

    #Assuming that is just one face on the image
    if len(detections) > 0:
        j = np.argmax(detections[0,0,:,2])
        confidence = detections[0,0,j,2]

        if confidence > 0.6:
        
            box=detections[0,0,j,3:7] * np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype("int")

            face = image[startY:endY,startX:endX]
            (fH,fW) = face.shape[:2]

            if fW < 20 or fH <20:
                continue
            

            #Every dict value has:
            # embeddings: The 128d vector with the face
            # name: Name or ID of this face
            # samples: No. of sample of this ID/name 
            if opencv:
                #Using openCV
                faceBlob = cv2.dnn.blobFromImage(face,1.0/255,(96,96),(0,0,0),swapRB=True,crop=False)
                emb.setInput(faceBlob)
                vec = emb.forward()
                knownNames.append(name)
                samples[name] = samples.get(name,0)+1
                knownEmbeddings.append(vec.flatten())
            else:    
                #Using dlib
                rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                locations = face_recognition.face_locations(rgb,model="cnn")
                encodings = face_recognition.face_encodings(rgb,locations,num_jitters=2,model="large")
                for enc in encodings:
                    knownEmbeddings.append(enc)
                    knownNames.append(name)
                    samples[name] = samples.get(name,0)+1

            bar.update(total+1)
            total+=1
bar.finish()

#Save embeddings dictionary  to the disk
print(len(knownEmbeddings))
print("[INFO] - Serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames, "samples": samples}
f = open("known/embeddings.pickle","wb")
f.write(pickle.dumps(data))
f.close()
