import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("clock.jpg", cv.IMREAD_GRAYSCALE)

gx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
gy = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)

# Compute the magnitude of the gradients
sobel = np.sqrt(gx**2 + gy**2)

# Normalize and threshold the magnitude image
edges = cv.normalize(sobel, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
edges = cv.threshold(edges, 30, 255, cv.THRESH_BINARY)[1]

plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(sobel, cmap = 'gray')
plt.title('sobel'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(edges, cmap = 'gray')
plt.title('edges'), plt.xticks([]), plt.yticks([])


linesList =[]
lines = cv.HoughLinesP(
            edges, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=90, # Min number of votes for valid line
            minLineLength=2, # Min allowed length of line
            maxLineGap=10 # Max allowed gap between line for joining them
            )
  
# Iterate over points
for points in lines:
      # Extracted points nested in the list
    x1,y1,x2,y2=points[0]
    # Draw the lines joing the points
    # On the original image
    cv.line(edges,(x1,y1),(x2,y2),255,1)
    # Maintain a simples lookup list for points
    linesList.append([(x1,y1),(x2,y2)])

plt.subplot(224),plt.imshow(edges, cmap = 'gray')
plt.title('hough transform'), plt.xticks([]), plt.yticks([])
plt.show()
