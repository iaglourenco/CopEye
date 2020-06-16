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
ap.add_argument("--model",help="Model to extract embeddings",required=False)
args = vars(ap.parse_args())



print("[INFO] - Loading face detector")
detector = cv2.dnn.readNetFromCaffe("models/face_detection_model/deploy.prototxt",
            "models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")


print("[INFO] - Loading faces")
imagePaths = list(paths.list_images("b99_cast"))
knownEmbeddings = []
knownNames = []
samples = {}

##Usar opencv para extrair embedding do rostos
if args.get("model") == None:
    opencv = False
else:
    opencv=True
    print("[INFO] - Loading embedding model")
    emb = cv2.dnn.readNetFromTorch(args["model"])

total = 0
bar = ProgressBar(maxval=len(imagePaths)).start()
bar.widgets.append(ETA())
bar.widgets.append(FileTransferSpeed(unit="img"))
for (i,imagePath) in enumerate(imagePaths):   
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
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

    detector.setInput(imageBlob)
    detections = detector.forward()

    if len(detections) > 0:
        j = np.argmax(detections[0,0,:,2])
        confidence = detections[0,0,j,2]
        if confidence > 0.5:
        
            box=detections[0,0,j,3:7] * np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype("int")

            face = image[startY:endY,startX:endX]
            (fH,fW) = face.shape[:2]

            if fW < 20 or fH <20:
                continue
            
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
                encodings = face_recognition.face_encodings(rgb,locations,model="large")
                for enc in encodings:
                    knownEmbeddings.append(enc)
                    knownNames.append(name)
                    samples[name] = samples.get(name,0)+1

            bar.update(total+1)
            total+=1
bar.finish()

print(len(knownEmbeddings))
print("[INFO] - Serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames, "samples": samples}
f = open("known/embeddings.pickle","wb")
f.write(pickle.dumps(data))
f.close()
