import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model=keras.models.load_model("Model/best.hdf5")

face_cascade=cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

video=cv2.VideoCapture(0)
classes=["Mask","NoMask"]
while True:
    ret,frame=video.read()
    faces=face_cascade.detectMultiScale(frame)
    for(x,y,w,h) in faces:
        roi=frame[y:y+h,x:x+w].copy()
        resized_img=cv2.resize(roi,(50,50))
        s=np.reshape(resized_img,(1,50,50,3))
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        print(classes[model.predict_classes(s)[0][0]])
        frame=cv2.putText(frame,classes[model.predict_classes(s)[0][0]],(x+10,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)  
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(15)
    if key==27:
        break
video.release()
cv2.destroyAllWindows()
