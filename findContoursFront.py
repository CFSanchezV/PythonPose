import numpy as np
import cv2
import os
import math

dirname = os.path.dirname(__file__)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    # calcs ratios
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)
    return resized


# Read image
img_in = cv2.imread(os.path.join(dirname, "filtered_images/result_front.jpg"))
img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

img_in = image_resize(img_in, height=730)
h, w  = img_in.shape

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
original = cv2.cvtColor(img_in, cv2.COLOR_GRAY2BGR)
canvas = np.zeros(original.shape, np.uint8)
img2gray = img_out.copy()

## blurring with kernel 25 ##
kernel = np.ones((5, 5), np.float32)/25
img2gray = cv2.filter2D(img2gray, -1, kernel)

#extract contours from thresholded image
_, thresh = cv2.threshold(img2gray, 250, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# find the main contour (2nd biggest area)
cnt = contours[-1]

# define main contour approx. and hull
# perimeter = cv2.arcLength(cnt, True)
epsilon = 0.01*cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

hull = cv2.convexHull(cnt)

# canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

# draw all points
cv2.drawContours(canvas, cnt, -1, (0, 255, 0), 2)

# draw approx, only some points using approxpolyDP
# cv2.drawContours(canvas, [approx], -1, (0, 0, 255), 2)

# cv2.drawContours(canvas, [hull], -1, (0, 0, 255), 3) # simple hull


# MEASUREMENTS AFTER CONTOURS ### NECK, WAIST, HIP and HEIGHT
neckY = 177
LwaistY, RwaistY = 306, 301
hipY = 396
footY = 647

def get_Xpts(contour, Ypoint):
    xs = []
    for point in contour:
        x, y = point[0][:]
        if y == Ypoint:
            xs.append( (x, y) )

    lst = list(set(xs))
    lst.sort()
    return lst

def show2points(lst):
    first, last = lst[0], lst[-1]
    cv2.circle(canvas, (first[0], first[1]), 4, (255,0,0), 2, cv2.LINE_AA)
    cv2.circle(canvas, (last[0], last[1]), 4, (255,0,0), 2, cv2.LINE_AA)

def showWaistpoints(LwaistPts, RwaistPts):
    middleX = w//2
    for pnt in LwaistPts:
        pass

    l1 = LwaistPts[0]
    r2 = RwaistPts[-1]
    cv2.circle(canvas, (l1[0], l1[1]), 4, (255,0,0), 2, cv2.LINE_AA)
    cv2.circle(canvas, (r2[0], r2[1]), 4, (255,0,0), 2, cv2.LINE_AA)


# Height calc
def get_TopY(contour, Xpoint):
    ys = []
    for point in contour:
        x, y = point[0][:]
        if x == Xpoint:
            ys.append( (x, y) )

    lst = list(set(ys))
    lst.sort()
    y = lst[0][1]

    return [Xpoint, y]


def calculate_Height(contour):
    midX = w//2
    bottomPoint = [midX, footY]
    topPoint = get_TopY(contour, midX)

    a = np.array(bottomPoint) # p1
    b = np.array(topPoint) # p2

    cv2.circle(canvas, a, 5, (255,0,0), 2)
    cv2.circle(canvas, b, 5, (255,0,0), 2)

    dist = np.linalg.norm(a - b)

    cv2.line(canvas, a, b, (255,0,0), 2, cv2.LINE_AA)

    return dist


person_height = int(calculate_Height(cnt))
print(person_height)


neckPts = get_Xpts(cnt, neckY)
LwaistPts = get_Xpts(cnt, LwaistY)
RwaistPts = get_Xpts(cnt, RwaistY)
hipPts = get_Xpts(cnt, hipY)


print(LwaistPts)
print(RwaistPts)

# DRAW POINTS
show2points(neckPts)
showWaistpoints(LwaistPts, RwaistPts)
show2points(hipPts)


cv2.imshow("Contours", canvas)
cv2.imwrite(os.path.join(dirname, 'filtered_images/result_frontPt2.jpg'), canvas)

if cv2.waitKey(0) & 0xFF == ord('q'): 
    cv2.destroyAllWindows()