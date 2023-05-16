import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("lenna.jpg", cv.IMREAD_GRAYSCALE)

def zonalMethod():
    n_horizontal = 8
    n_vertical = 8

    # Create a mask for the selected coefficients
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[:n_vertical, :n_horizontal] = 1
    return mask
def thresholdMethod(img):
    mask = np.zeros((32, 32), dtype=np.uint8)
    for i in range(32):
        for j in range(32):
            if img[i][j]>125:
                mask[i][j]=1
    return mask
    
    
height  = len(img) #one column of image
width = len(img[0]) # one row of image
sliced = [] # new list for 32x32 sliced image 
block = 32


#dividing 32x32 parts
currY = 0 #current Y index
for i in range(block,height+1,block):
    currX = 0 #current X index
    for j in range(block,width+1,block):
        sliced.append(img[currY:i,currX:j]-np.ones((32,32))*128) #Extracting 128 from all pixels
        currX = j
    currY = i
    
imf = [np.float32(img) for img in sliced]
DCToutput = []
for part in imf:
    currDCT = cv.dct(part)
    DCToutput.append(currDCT)
DCToutput[0][0]

selectedQMatrix = zonalMethod()
selectedQMatrix = thresholdMethod(img)

for ndct in DCToutput:
    for i in range(block):
        for j in range(block):
            ndct[i,j] = np.around(ndct[i,j]*selectedQMatrix[i,j])
DCToutput[0][0]

invList = []
for ipart in DCToutput:
    ipart
    curriDCT = cv.idct(ipart)
    invList.append(curriDCT)
invList[0][0]

row = 0
rowNcol = []
for j in range(int(width/block),len(invList)+1,int(width/block)):
    rowNcol.append(np.hstack((invList[row:j])))
    row = j
res = np.vstack((rowNcol))


plt.subplot(211),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(res, cmap = 'gray')
#plt.title('compressed (zonal method)'), plt.xticks([]), plt.yticks([])
plt.title('compressed (threshold method)'), plt.xticks([]), plt.yticks([])
plt.show()