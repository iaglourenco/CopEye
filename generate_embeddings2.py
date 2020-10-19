# Generate the embeddings using the functions implemented in functions.py


from imutils import paths
import os
import argparse
from functions import *

ap = argparse.ArgumentParser()
ap.add_argument("--data",help="Path to the dataset",required=True)
args = vars(ap.parse_args())



## Get the images path
print("[INFO] - Loading images")
imagePaths = list(paths.list_images(args.get("data")))
align_faces(imagePaths)
print("[INFO] - {} images loaded".format(len(imagePaths)))
names = []


try:
    #For each image path load the image and extract the face embedding
    for (i,imagePath) in enumerate(imagePaths):   
        names.append(imagePath.split(os.path.sep)[-2])        
    
    update_db_encodings(names,imagePaths)


except KeyboardInterrupt:
    print("\n[INFO] - Stopped by user")
   
