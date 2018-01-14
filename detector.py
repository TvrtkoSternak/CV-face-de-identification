import numpy as np
import cv2

class Detector:
    def __init__(self):
        haar_cascade_front = 'Classifiers/haarcascade_frontalface_default.xml'
        haar_cascade_profile = 'Classifiers/haarcascade_profileface.xml'

        self.frontal_face_cascade = cv2.CascadeClassifier(haar_cascade_front)
        self.profile_face_cascade = cv2.CascadeClassifier(haar_cascade_profile)

    def detect(self, image_file):

        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rectangles = []
        faces_front = self.frontal_face_cascade.detectMultiScale(gray, 1.3, 5)
        faces_profile = self.profile_face_cascade.detectMultiScale(gray, 1.3, 5)
        rectangles.extend(faces_front)
        rectangles.extend(faces_profile)
        return rectangles

    def show_on_image(self, image_file, rectangles):

        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for (x,y,w,h) in rectangles:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

        return img


if __name__ == '__main__':
    image = 'Images/people-02.jpg'
    dt = Detector()
    recs = dt.detect(image)
    img = dt.show_on_image(image, recs)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
