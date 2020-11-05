# Align and resize faces by the eyes and head, using imutils
# I'm assuming that have only one face in the images
#
# @idea: Adrian Rosebrock - PyImageSearch.com
# @author: Iago Lourenço - @iaglourenco
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import paths
import imutils
import dlib
import cv2
import pickle
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",help="dataset to use",required=True)
args = vars(ap.parse_args())

print("[INFO] - Loading images")
imagePaths = list(paths.list_images(args["dataset"]))
print("[INFO] - Loaded {} images".format(len(imagePaths)))


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../one_shot/models/shape_predictor_68_face_landmarks.dat")
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