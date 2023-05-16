import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


dr1 = cv.imread('dark_road_1.jpg', cv.IMREAD_GRAYSCALE)
dr2 = cv.imread('dark_road_2.jpg', cv.IMREAD_GRAYSCALE)
dr3 = cv.imread('dark_road_3.jpg', cv.IMREAD_GRAYSCALE)

hist1 = cv.calcHist([dr1],[0],None,[256],[0,256])
hist2 = cv.calcHist([dr2],[0],None,[256],[0,256])
hist3 = cv.calcHist([dr3],[0],None,[256],[0,256])


eqDr1 = cv.equalizeHist(dr1)
eqDr2 = cv.equalizeHist(dr2)
eqDr3 = cv.equalizeHist(dr3)



eqHist1 = cv.calcHist([eqDr1], [0],None,[256],[0,256]) #For some reason many values are zero here
eqHist2 = cv.calcHist([eqDr2], [0],None,[256],[0,256])
eqHist3 = cv.calcHist([eqDr3], [0],None,[256],[0,256])


clahe = cv.createCLAHE(clipLimit=6.0, tileGridSize=(16,16))


adEqDr1 = clahe.apply(dr1)
adEqDr2 = clahe.apply(dr2)
adEqDr3 = clahe.apply(dr3)


adEqHist1 = cv.calcHist([adEqDr1], [0],None,[256],[0,256]) #For some reason many values are zero here
adEqHist2 = cv.calcHist([adEqDr2], [0],None,[256],[0,256])
adEqHist3 = cv.calcHist([adEqDr3], [0],None,[256],[0,256])


plt.subplot(231),plt.imshow(dr1, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(dr2, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(dr3, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.hist(hist1.ravel(),256,[0,256]) 
plt.subplot(235), plt.hist(hist2.ravel(),256,[0,256])
plt.subplot(236), plt.hist(hist3.ravel(),256,[0,256])
plt.show()

plt.subplot(231),plt.imshow(eqDr1, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(eqDr2, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(eqDr3, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(234), plt.hist(eqHist1.ravel(),256,[0,256]) 
plt.ylim([0, 10])
plt.subplot(235), plt.hist(eqHist2.ravel(),256,[0,256])
plt.ylim([0, 10])
plt.subplot(236), plt.hist(eqHist3.ravel(),256,[0,256])
plt.ylim([0, 10])
plt.show()

plt.subplot(231),plt.imshow(adEqDr1, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(adEqDr2, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(adEqDr3, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.hist(adEqHist1.ravel(),256,[0,256]) 
plt.subplot(235), plt.hist(adEqHist2.ravel(),256,[0,256])
plt.subplot(236), plt.hist(adEqHist3.ravel(),256,[0,256])

plt.show()