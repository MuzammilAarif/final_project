import warnings
import joblib
warnings.filterwarnings("ignore")

model1=joblib.load('model1.pkl')

import cv2
import sys
from PIL import Image

cascPath = sys.argv[1]
#faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)
condition = True

while condition:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
        
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    k = cv2.waitKey(100)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('result/img1.jpg',frame)
        cv2.destroyAllWindows()
        condition = False
    
video_capture.release()
cv2.destroyAllWindows()

import os
import cv2

IMG_DIR = 'result'

for img in os.listdir(IMG_DIR):
        img_array = cv2.imread(os.path.join(IMG_DIR,img), cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array,(30,30))

        img_array = (img_array.flatten())

        img_array  = img_array.reshape(-1, 1).T

        print(img_array)
        #with open('result1.csv', 'ab') as f:
            #np.savetxt(f,img_array, delimiter=",")

import numpy

img_array=numpy.array(img_array)

print(img_array)

type(img_array)

pre=model1.predict(img_array)

pre=pre[0]

pre



##Serial communication

import serial

ser = serial.Serial()
ser.port = 'COM3'
ser.baudrate = 9600
ser.bytesize = 8
ser.parity = serial.PARITY_NONE
ser.stopbits = serial.STOPBITS_ONE
ser.open()
#a = b'*R#'
#ser.write(a)
if pre == 'person_a':
    v=b'*1#'
    ser.write(v)
else:
    v=b'*2#'
    ser.write(v)