# Generate the embeddings using the functions implemented in functions.py

import random
from imutils import paths
import os
import argparse
from functions import *
import database

ap = argparse.ArgumentParser()
ap.add_argument("--data",help="Path to the dataset",required=True)
args = vars(ap.parse_args())


## Get the images path
print("[INFO] - Loading images")
imagePaths = list(paths.list_images(args.get("data")))
# align_faces(imagePaths)
print("[INFO] - {} images loaded".format(len(imagePaths)))
names = []
n ="nome"
last_n = ''
perigos = ['Alto','Medio','Baixo']


try:
    print("[INFO] - Inserting data to default database, please wait...")
    #For each image path load the image and extract the face embedding
    for (i,imagePath) in enumerate(imagePaths):   
        n = imagePath.split(os.path.sep)[-2]
        names.append(imagePath.split(os.path.sep)[-2])        
        

        if not n == last_n:
            last_n = n
            age = random.randint(20,40)
            perigo = random.choice(perigos)

        sqlite_add_fugitives(defaultdb,database.Fugitivo(n,age,perigo),[imagePath],artigo=157)

    # update_db_encodings(names,imagePaths)


except KeyboardInterrupt:
    print("\n[INFO] - Stopped by user")
   
