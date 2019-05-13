#import liberaries
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from keras.preprocessing.image import img_to_array

#use the haarcascade frontal face detection and miniXception trained model for emotion detection
detection_model_path = '.\\haarcascade_frontalface_default.xml'
emotion_model_path = '.\\.19-0.65.hdf5'

#label dataset
def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
        
emotion_labels = get_labels('fer2013')        
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

face_classifier = cv2.CascadeClassifier('.\\haarcascade_frontalface_alt.xml')

emotion_classifier = load_model(emotion_model_path, compile=False)

emotion_target_size = emotion_classifier.input_shape[1:3]

#read each frame from camera and separate the no of face present in that and use classifier to predict the emotion
def face_detector(img):
    # Convert image to grayscale
    imgc=img
    gray = cv2.cvtColor(imgc,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.1, 5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    if faces is ():
        return (0,0,0,0), np.zeros((48,48), np.uint8), img
    
    allfaces = []   
    rects = []
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
        allfaces.append(roi_gray)
        rects.append((x,w,y,h))
    return rects, allfaces, img

# use camera functionality in real time for classification of faces
def main():
    cap=cv2.VideoCapture(0)
    while (cap.isOpened()):
        
        ret,img=cap.read()
        rects, faces, image = face_detector(img)
        
        
        #img = cv2.imread("Akshay.jpg")
        #rects, faces, image = face_detector(img)
        
        i = 0
        for face in faces:
            roi = face.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            emotion_label_arg = np.argmax(emotion_classifier.predict(roi)[0])
            emotion_text = emotion_labels[emotion_label_arg]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,emotion_text,(100,200), font, 4,(255,255,255),2,cv2.LINE_AA)
            print(emotion_text)
            cv2.imshow('result',img)
        
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break    
       
        
main()
