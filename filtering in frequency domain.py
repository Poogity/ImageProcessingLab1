import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("aerial.tiff", cv.IMREAD_GRAYSCALE)

imgFreq = np.fft.fft2(img)
imgFreq = np.fft.fftshift(imgFreq)

lpfKernel = np.ones((5,5),np.float32)/25
lpfKernel = np.array([[1,2,3,2,1],
                      [2,3,5,3,2],
                      [3,5,10,5,3],
                      [2,3,5,3,2],
                      [1,2,3,2,1]])/74
smoothedImg = cv.filter2D(img,-1,lpfKernel)

hpfKernel = np.array([[-1/30,-1/30,-1/30,-1/30,-1/30],
                     [-1/30,-1/30,-1/30,-1/30,-1/30],
                     [-1/30,-1/30, 29/30,-1/30,-1/30],
                     [-1/30,-1/30,-1/30,-1/30,-1/30],
                     [-1/30,-1/30,-1/30,-1/30,-1/30]])


sharpenedImg = cv.filter2D(img,-1,hpfKernel)

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(20*np.log(abs(imgFreq)), cmap = 'gray')
plt.title('Frequency Domain (dB)'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(221),plt.imshow(lpfKernel, cmap = 'gray')
plt.title('LPF Impulse'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(hpfKernel, cmap = 'gray')
plt.title('HPF Impulse'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(20*np.log(abs(np.fft.fftshift(np.fft.fft2(smoothedImg)/imgFreq))), cmap = 'gray')
plt.title('LPF Magnitude (dB)'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(20*np.log(abs(np.fft.fftshift(np.fft.fft2(sharpenedImg)/imgFreq))), cmap = 'gray')
plt.title('HPF Magnitude (dB)'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(311),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(312),plt.imshow(smoothedImg, cmap = 'gray')
plt.title('Smoothed Image (LPF)'), plt.xticks([]), plt.yticks([])
plt.subplot(313),plt.imshow(sharpenedImg, cmap = 'gray')
plt.title('Sharpened Image (HPF)'), plt.xticks([]), plt.yticks([])
plt.show()