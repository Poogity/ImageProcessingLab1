import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random

def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col= image.shape
      mean = 0
      var = image.var()/10**0.75 #SNR = 20log(signal_variance/noise_variance) = 15dB
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col))
      gauss = gauss.reshape(row,col)
      noisy = image + gauss
      return noisy
  
   elif noise_typ == "s&p":
          # Getting the dimensions of the image
        noisy = image.copy()
        row , col = noisy.shape
          
       
        number_of_pixels = int(row*col/8)
        for i in range(number_of_pixels):
            
            # Pick a random y coordinate
            y_coord=random.randint(0, row - 1)
              
            # Pick a random x coordinate
            x_coord=random.randint(0, col - 1)
              
            # Color that pixel to white
            noisy[y_coord][x_coord] = 255
              
        
        number_of_pixels = int(row*col/8)
        for i in range(number_of_pixels):
            
            # Pick a random y coordinate
            y_coord=random.randint(0, row - 1)
              
            # Pick a random x coordinate
            x_coord=random.randint(0, col - 1)
              
            # Color that pixel to black
            noisy[y_coord][x_coord] = 0
              
        return noisy


img = cv.imread("flower.png", cv.IMREAD_GRAYSCALE)
noisyImg = noisy("s&p",img)
movingAv = cv.GaussianBlur(noisyImg,(5,5),0)
median = cv.medianBlur(noisyImg.astype('float32'), 5)

plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(noisyImg, cmap = 'gray')
#plt.title('White Noise'), plt.xticks([]), plt.yticks([])
#plt.subplot(223),plt.imshow(movingAv, cmap = 'gray')
plt.title('Impulse Noise'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(movingAv, cmap = 'gray')
plt.title('Moving Average Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(median, cmap = 'gray')
plt.title('median Filter'), plt.xticks([]), plt.yticks([])
plt.show()