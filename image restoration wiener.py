import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


  
img = cv.imread("factory.jpg", cv.IMREAD_GRAYSCALE)

#smoothing
gaussKernel = cv.getGaussianKernel(5, 2)
noisyImg = cv.filter2D(img,-1,gaussKernel).astype('float32')

#adding noise
sigma = np.sqrt(noisyImg.var()/10**0.5)
gaussNoise = np.random.normal(0,sigma,img.shape)
noisyImg += gaussNoise

# Compute the Wiener filter and denoise the image
psf = np.abs(np.fft.fft2(noisyImg)) ** 2
transferFunction = np.conj(np.fft.fft2(gaussKernel, noisyImg.shape[:2]))
wienerFilter = transferFunction / (transferFunction**2 + sigma ** 2 / psf)
restoredImg = np.fft.ifft2(wienerFilter * np.fft.fft2(noisyImg)).real

# Compute the inverse Wiener filter and remove the convolutional degrading
inverseFilter = 1 / wienerFilter
restoredImg = np.fft.ifft2(inverseFilter * np.fft.fft2(noisyImg)).real


plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(noisyImg, cmap = 'gray')
plt.title('Noisy Image'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(restoredImg, cmap = 'gray')
plt.title('Wiener Filtered'), plt.xticks([]), plt.yticks([])


snr = 10.0  # Signal-to-noise ratio
psf = cv.getGaussianKernel(21, 1.7) * cv.getGaussianKernel(21, 1.7).T
k = snr / (snr + 1.0 / np.mean(psf))  # Wiener filter balance parameter
wienerFilter = (np.conj(psf) / np.abs(psf)**2) * (np.abs(psf)**2 / (np.abs(psf)**2 + k))

# Perform the Wiener deconvolution
deconvolvedImg = cv.filter2D(noisyImg.astype(np.float32), -1, wienerFilter)

plt.subplot(224),plt.imshow(deconvolvedImg, cmap='gray')
plt.title('Wiener Deconvolved'), plt.xticks([]), plt.yticks([])
plt.show()