import numpy as np
import cv2
import os

dirname = os.path.dirname(__file__)

# Read image
img_in = cv2.imread(os.path.join(dirname, "filtered_images/result.jpg"))
img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

# Show B&W image
# cv2.imshow("Input img", img_in)
 
# Threshold. Set values equal to or above 220 to 0. Set values below 220 to 255.
th, img_thres = cv2.threshold(img_in, 245, 255, cv2.THRESH_BINARY_INV)
 
# Copy the thresholded image.
img_floodfill = img_thres.copy()
 
# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = img_thres.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(img_floodfill, mask, (0,0), 255)

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(img_floodfill)

# Combine the two images to get the foreground.
img_out = img_thres | im_floodfill_inv

# im_out is of shape (w, h, _)

# Simulate receiving countour processed image 
# cv2.imshow('floodFilledImg', img_out)

# get a blank canvas for drawing contour on and convert img to grayscale
img_in = cv2.cvtColor(img_in, cv2.COLOR_GRAY2BGR)
canvas = np.zeros(img_in.shape, np.uint8)
img2gray = img_out.copy()

## blurring with kernel 25 ##
kernel = np.ones((5, 5), np.float32)/25
img2gray = cv2.filter2D(img2gray, -1, kernel)

#extract contours from thresholded image
_, thresh = cv2.threshold(img2gray, 250, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# find the main contour (2nd biggest area)
cnt = contours[-1]
print(len(cnt))

# define main contour approx. and hull
perimeter = cv2.arcLength(cnt, True)
epsilon = 0.01*cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

hull = cv2.convexHull(cnt)

# canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

# draw all points
cv2.drawContours(canvas, cnt, -1, (0, 255, 0), 2)
# draw some points by approxpolyDP
cv2.drawContours(canvas, [approx], -1, (0, 0, 255), 2)

# cv2.drawContours(canvas, [hull], -1, (0, 0, 255), 3) # simple hull

cv2.imshow("Contours", canvas)
# cv2.imwrite(os.path.join(dirname, 'filtered_images/resultPt2.jpg'), canvas)

if cv2.waitKey(0) & 0xFF == ord('q'): 
    cv2.destroyAllWindows()