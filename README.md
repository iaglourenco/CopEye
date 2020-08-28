# CopEye

![alt](https://github.com/iaglourenco/CopEye/blob/develop/examples/videogif.gif?raw=true)

![alt](https://github.com/iaglourenco/CopEye/blob/develop/examples/recog.png?raw=true)
![alt](https://github.com/iaglourenco/CopEye/blob/develop/examples/compare1.png?raw=true)
![alt](https://github.com/iaglourenco/CopEye/blob/develop/examples/compare2.png?raw=true)

## How it works

Using the One-shot technique, it recognizes the person with just one photo.

```Using more photos it will only improve the result, but try to vary the poses```

## How to execute

Be sure to have python,dlib and opencv installed (there's more modules needed)

There's a precompiled dataset on the repo, it have more than 100 celebrities registered, you can run the script directly with `one_shot.py` and see the algorithm find the celebrity most like you or try it in a movie trailer(check [input](https://github.com/iaglourenco/CopEye/tree/master/input) folder for a surprise :wink:) with `one_shot_video.py` (*run `<script-name> -h` for how-to-use*)

If you want to run in a new dataset, do this:

1. Put your photos in a folder in /datasets
2. Run `align_faces.py` in /tools passing the path of the photos you just added (this step is for better results)
3. Run `generate_embeddings.py` to generate the embeddings (*obviously*)
4. Now is the time to play! :smile:, run `one_shot.py` for real-time recognition using your webcam or `one_shot_video.py` for recognize persons in a video file (*only supports mp4 videos*)

## Need help?
All scripts have a `-h` option that will explain what you need to pass in arguments.

And you can contact me btw!

## Beautiful and wonderful projects and software that were used here  

[dlib](http://dlib.net/)

[OpenCV](https://opencv.org/)

[face_recognition](https://github.com/ageitgey/face_recognition)

[Pins Face Recognition Dataset](https://www.kaggle.com/hereisburak/pins-face-recognition/data)

[OpenFace](https://cmusatyalab.github.io/openface/)