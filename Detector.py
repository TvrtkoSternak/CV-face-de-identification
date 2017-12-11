import numpy as np
import cv2

frontal_face_cascade = cv2.CascadeClassifier('Classifiers/haarcascade_frontalface_default.xml')
profile_face_cascade = cv2.CascadeClassifier('Classifiers/haarcascade_profileface.xml')

img = cv2.imread('Images/people-02.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = frontal_face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

faces = profile_face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()