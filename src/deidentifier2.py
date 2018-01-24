from scipy import fftpack
import numpy

class Deidentifier:
    def __init__(self):
        pass

    def gaussian_kernel(self, size, size_y=None):
        if not size_y:
            size_y = size
        x, y = numpy.mgrid[0:size, 0:size_y]
        if size < 100: sigma = 10
        elif size < 200: sigma = 13
        else: sigma = 16
        g = numpy.exp(-(x ** 2 + y ** 2) / (sigma ** 2))
        return g / g.sum()

    def deidentify(self, img):
        gaussian = self.gaussian_kernel(img.shape[0]);
        gaussian_ft = fftpack.fft2(gaussian, axes=(0, 1))
        img_ft = fftpack.fft2(img, axes=(0, 1))
        # convolution by Fourier
        image_blurred_ft = gaussian_ft[:, :] * img_ft
        image_blurred = fftpack.ifft2(image_blurred_ft, axes=(0, 1)).real
        return image_blurred

    def identify(self, img):
        image_blurred_ft = fftpack.fft2(img, axes=(0, 1))
        gaussian = self.gaussian_kernel(img.shape[0])
        gaussian_ft = fftpack.fft2(gaussian, axes=(0, 1))
        # deconvolution by fourier transform
        image_deblurred_ft = image_blurred_ft / gaussian_ft[:, :]
        image_deblurred = fftpack.ifft2(image_deblurred_ft, axes=(0, 1)).real
        return image_deblurred
