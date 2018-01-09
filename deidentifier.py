
from scipy.ndimage import gaussian_filter
import cv2

class Deidentifier:
    def __init__(self):
        pass
    
    def deidentify(self, image_file, rectangles=None):
        img = cv2.imread(image_file)
        image_blurred = img
        if rectangles:
            for (x, y, w, h) in rectangles:
                image_blurred[y:y+h, x:x+w] = gaussian_filter(image_blurred[y:y+h, x:x+w], 5)
        else:
            image_blurred = gaussian_filter(img, 5)

        return image_blurred


if __name__ == '__main__':
    di = Deidentifier()
    img = di.deidentify('Images/people-02.jpg')
    cv2.imshow('output', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
