#!/usr/bin/python3
from tensorflow import keras
import cv2
import os
import numpy as np
emotion=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


model=keras.models.load_model("model2.h5")
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        crop_image=gray[y:y+h,x:x+w]
        resize_image=cv2.resize(crop_image,(48,48))
        resize_image=resize_image.astype(np.float32)
        resize_image=resize_image/255
        resize_image=np.expand_dims(resize_image,axis=0)
        reshape_image=resize_image.reshape(1,48,48,1)
        predict_vector=model.predict(resize_image)[0]
        index=np.where(predict_vector == np.amax(predict_vector))
        text = emotion[int(index[0])] 
        coordinates = (y+10,x+10)                                                                              
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX                                                                  
        fontScale = 1                                                                                    
        color = (0,0,255)                                                                              
        thickness = 1                                                                                    
        frames = cv2.putText(frames, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)    
    
    
    # Display the resulting frame
    cv2.imshow('Video', frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
