from scipy import fftpack, stats
import numpy
import matplotlib.pyplot as plt
from PIL import Image

class Deidentifier:
    def __init__(self):
        pass

    def gaussian_kernel(self, kernlen=43, nsig=3):
        interval = (2 * nsig + 1.) / (kernlen)
        x = numpy.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = numpy.diff(stats.norm.cdf(x))
        kernel_raw = numpy.sqrt(numpy.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
    
    def deidentify(self, image_file, rectangles=None):
        img2 = Image.open(image_file)
        # converting to png
        img2.save("Images/people-test.png")
        img = plt.imread("Images/people-test.png")
        gaussian = self.gaussian_kernel()
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
        gaussian = self.gaussian_kernel()
        gaussian_ft = fftpack.fft2(gaussian, shape=img.shape[:2], axes=(0, 1))
        # deconvolution by fourier transform
        image_deblurred_ft = image_blurred_ft / gaussian_ft[:, :, numpy.newaxis]
        image_deblurred = fftpack.ifft2(image_deblurred_ft, axes=(0, 1))
        image_deblurred = numpy.clip(image_deblurred, 0, 1)
        return image_deblurred

if __name__ == '__main__':
    di = Deidentifier()
    img = di.deidentify('Images/people-01.jpg')
    plt.figure()
    plt.imshow(img.real)
    deblurred = di.identify(img)
    plt.figure()
    plt.imshow(deblurred.real)
    plt.show()
