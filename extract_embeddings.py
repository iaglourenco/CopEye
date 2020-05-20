from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os
from progressbar import ProgressBar,ETA,FileTransferSpeed
import time


print("[INFO] - Loading face detector")
detector = cv2.dnn.readNetFromCaffe("models/face_detection_model/deploy.prototxt",
            "models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")

print("[INFO] - Loading embedding model")
emb = cv2.dnn.readNetFromTorch("models/nn4.v2.t7")

print("[INFO] - Loading faces")
imagePaths = list(paths.list_images("dataset"))
knownEmbeddings = []
knownNames = []

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
        if confidence > 0.6:
        
            box=detections[0,0,j,3:7] * np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype("int")

            face = image[startY:endY,startX:endX]
            (fH,fW) = face.shape[:2]

            if fW < 20 or fH <20:
                continue
         
            faceBlob = cv2.dnn.blobFromImage(face,1.0/255,(96,96),(0,0,0),swapRB=True,crop=False)
            emb.setInput(faceBlob)
            vec = emb.forward()
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            
            
           
            bar.update(total+1)
            total+=1
bar.finish()

print("[INFO] - Serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open("pickle/embeddings.pickle","wb")
f.write(pickle.dumps(data))
f.close()