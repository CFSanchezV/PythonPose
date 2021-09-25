import numpy as np
import cv2
import os

dirname = os.path.dirname(__file__)
font = cv2.FONT_HERSHEY_COMPLEX 

# Simulate receiving countour processed image
image = cv2.imread(os.path.join(dirname, '../input/t-pose.jpg')) 
cv2.imshow('Imagen inicial', image)

# get a blank canvas for drawing contour on and convert img to grayscale
canvas = np.zeros(image.shape, np.uint8)
img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

## blurring with kernel 25 ##
kernel = np.ones((5, 5), np.float32)/25
img2gray = cv2.filter2D(img2gray, -1, kernel)
# cv2.imshow('Imagen post kernel', img2gray)

# threshold the image and extract contours
ret, thresh = cv2.threshold(img2gray, 250, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# find the main contour (2nd biggest area)
cnt = contours[0]
print(len(contours))

# define main island contour approx. and hull
perimeter = cv2.arcLength(cnt, True)
epsilon = 0.01*cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

hull = cv2.convexHull(cnt)

# draw all points
cv2.drawContours(canvas, cnt, -1, (0, 255, 0), 2)
# draw some points by approxpolyDP
cv2.drawContours(canvas, approx, -1, (0, 0, 255), 2)

## cv2.drawContours(canvas, hull, -1, (0, 0, 255), 3) # only displays 

cv2.imshow("Contour", canvas)

if cv2.waitKey(0) & 0xFF == ord('q'): 
    cv2.destroyAllWindows()