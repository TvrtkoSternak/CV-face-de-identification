#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import detector
import deidentifier

if __name__ == '__main__':
    image = 'Images/people-02.jpg'
    dt = detector.Detector()
    di = deidentifier.Deidentifier()

    recs = dt.detect(image)
    blur = di.deidentify(image, recs)
    faces = dt.show_on_image(image, recs)

    cv2.imshow('faces', faces)
    cv2.imshow('blur', blur)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
