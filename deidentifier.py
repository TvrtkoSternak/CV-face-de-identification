from scipy import fftpack
import cv2, numpy
import matplotlib.pyplot as plt

class Deidentifier:
    def __init__(self):
        pass

    def gaussian_kernel(self, size, size_y=None):
        size = int(size)
        if not size_y:
            size_y = size
        else:
            size_y = int(size_y)
        x, y = numpy.mgrid[-size:size + 1, -size_y:size_y + 1]
        g = numpy.exp(-(x ** 2 / float(size) + y ** 2 / float(size_y)))
        return g / g.sum()
    
    def deidentify(self, image_file, rectangles=None):
        img = plt.imread(image_file)
        gaussian = self.gaussian_kernel(10)
        gaussian_ft = fftpack.fft2(gaussian, shape=img.shape[:2], axes=(0, 1))
        img_ft = fftpack.fft2(img, axes=(0, 1))
        if rectangles:
            for (x, y, w, h) in rectangles:
                image_blurred_ft = img_ft
                image_blurred_ft[y:y+h, x:x+w] = gaussian_ft[:, :, numpy.newaxis] * img_ft
        else:
            image_blurred_ft = gaussian_ft[:, :, numpy.newaxis] * img_ft

        image_blurred = fftpack.ifft2(image_blurred_ft, axes=(0, 1))
        # clip values to range
        image_blurred = numpy.clip(image_blurred, 0, 1)
        return image_blurred

    def identify(self, img):
        image_blurred_ft = fftpack.fft2(img, axes=(0, 1))
        gaussian = self.gaussian_kernel(10)
        gaussian_ft = fftpack.fft2(gaussian, shape=img.shape[:2], axes=(0, 1))
        image_deblurred_ft = image_blurred_ft / gaussian_ft[:, :, numpy.newaxis]
        image_deblurred = fftpack.ifft2(image_deblurred_ft, axes=(0, 1))
        image_deblurred = numpy.clip(image_deblurred, 0, 1)
        return image_deblurred

if __name__ == '__main__':
    di = Deidentifier()
    img = di.deidentify('Images/people-01.png')
    plt.figure()
    plt.imshow(img.real)
    deblurred = di.identify(img)
    plt.figure()
    plt.imshow(deblurred.real)
    plt.show()
