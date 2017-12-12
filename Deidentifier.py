from scipy.ndimage import gaussian_filter
import cv2
img = cv2.imread('Images/people-02.jpg')
output = gaussian_filter(img, 5)
cv2.imshow('output',output)
cv2.waitKey(0)