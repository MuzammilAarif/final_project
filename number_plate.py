import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import re


import cv2
import numpy as np

numberPlateCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml') 
plat_detector =  cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
img = cv2.imread("/home/ima-u/Project-C/Aarif/Final_Project/IMA_final/data/img8.webp")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plates = plat_detector.detectMultiScale(img,scaleFactor=1.2,minNeighbors = 5, minSize=(25,25))   

for (x,y,w,h) in plates:
    cv2.putText(img,text='License Plate',org=(x-3,y-3),fontFace=cv2.FONT_HERSHEY_COMPLEX,color=(0,0,255),thickness=1,fontScale=0.6)
    #img[y:y+h,x:x+w] = cv2.blur(img[y:y+h,x:x+w],ksize=(10,10))
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


#plt.imshow(img)

from PIL import Image

img1=Image.fromarray(img)

img1.save('new1.jpg')



## 1. Read in Image, Grayscale and Blur
img = cv2.imread('new1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))


## 2. Apply filter and find edges for localization
bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
edged = cv2.Canny(bfilter, 30, 200) #Edge detection
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))


## 3. Find Contours and Apply Mask
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 20, True)
    if len(approx) == 4:
        location = approx
        break

location

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

# cv2.imshow("Result", new_image)

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))


## 4. Use Easy OCR To Read Text
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
result


## 5. Render Result
text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))

#type(text)

print(text)

text = "".join(re.split("[^a-zA-Z0-9]*", text))

print("Number_Plate: ",text)

# ##Serial communication
# import serial
# import time
# ser = serial.Serial()
# ser.port = 'COM4'
# ser.baudrate = 9600
# ser.bytesize = 8
# ser.parity = serial.PARITY_NONE
# ser.stopbits = serial.STOPBITS_ONE
# ser.open()
# a = b'*R#'
# ser.write(a)
# time.sleep(5)
# if text == 'HRZ6DK8337':
#     v=b'*1#'
#     ser.write(v)
# else:
#     v=b'*2#'
#     ser.write(v)