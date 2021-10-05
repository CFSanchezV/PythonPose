import numpy as np
import cv2
import os

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
# h, w  = img_in.shape

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
print("F Con height:", h)
print("F Con width:", w)

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
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# find the main contour (2nd biggest area)
def findMainContour(contours):
    areas = []
    for cont in contours:
        areas.append(cv2.contourArea(cont))

    n= len(areas)
    areas.sort()
    return contours[areas.index(areas[n-1])]

cnt = contours[-1]
if len(contours) != 1:
    cnt = findMainContour(contours)

# define main contour approx. and hull
# perim = cv2.arcLength(cnt, True)
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
dist_chest_front = 119.340

def get_Xpts(contour, Ypoint):
    xs = []
    for point in contour:
        x, y = point[0][:]
        if y == Ypoint:
            xs.append( (x, y) )

    lst = list(set(xs))
    lst.sort()
    return lst

def get2Points(lst):
    first, last = lst[0], lst[-1]
    cv2.circle(canvas, (first[0], first[1]), 4, (255,0,0), 2, cv2.LINE_AA)
    cv2.circle(canvas, (last[0], last[1]), 4, (255,0,0), 2, cv2.LINE_AA)

    first, last = [first[0], first[1]] , [last[0], last[1]]
    return first, last

def getWaistPoints(LwaistPts, RwaistPts):
    mid = w//2 # mid
    minLDist = w # max length
    minRDist = w # max length
    Lpoint, Rpoint = (0, 0), (0,0)

    for pnt in LwaistPts:
        x, y = pnt[:] # point
        if x > mid: continue
        dist = abs(mid - x)
        if dist < minLDist:
            minLDist = dist
            Lpoint = (x, y)

    for pnt in RwaistPts:
        x, y = pnt[:] # point
        if x < mid: continue
        dist = abs(mid - x)
        if dist < minRDist:
            minRDist = dist
            Rpoint = (x, y)
    
    cv2.circle(canvas, Rpoint, 4, (255,0,0), 2, cv2.LINE_AA)
    cv2.circle(canvas, Lpoint, 4, (255,0,0), 2, cv2.LINE_AA)

    waistPt1, waistPt2 = [Rpoint[0], Rpoint[1]] , [Lpoint[0], Lpoint[1]]
    return waistPt1, waistPt2


### Height calc ###
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
    # cv2.line(canvas, a, b, (255,0,0), 2, cv2.LINE_AA)

    return dist


person_height = calculate_Height(cnt)


### misc calcs ###
neckPts = get_Xpts(cnt, neckY)
hipPts = get_Xpts(cnt, hipY)

LwaistPts = get_Xpts(cnt, LwaistY)
RwaistPts = get_Xpts(cnt, RwaistY)


# DRAW POINTS
FneckPt1, FneckPt2 = get2Points(neckPts)
FwaistPt1, FwaistPt2 = getWaistPoints(LwaistPts, RwaistPts)
FhipPt1, FhipPt2 = get2Points(hipPts)



### CALCULATE SIZES ###
def calculate_Distance(pt1, pt2):
    '''makes numpy [x y] array from an [x,y] list to calc dist'''
    a = np.array(pt1) # p1
    b = np.array(pt2) # p2

    dist = np.linalg.norm(a - b)

    return dist


### SHOW FRONT RESULTS ###
distNeck = calculate_Distance(FneckPt1, FneckPt2)
distWaist = calculate_Distance(FwaistPt1, FwaistPt2)
distHip = calculate_Distance(FhipPt1, FhipPt2)

topL1, topL2, topL3, topL4, topL5 = (0,15), (0,30), (0,45), (0,60), (0,75)
cv2.putText(canvas, "neck: {:.2f}".format(distNeck), topL1, 
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

cv2.putText(canvas, "chest: {:.2f}".format(dist_chest_front), topL2, 
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

cv2.putText(canvas, "waist: {:.2f}".format(distWaist), topL3, 
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

cv2.putText(canvas, "hip: {:.2f}".format(distHip), topL4, 
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

cv2.putText(canvas, "height: {:.2f}".format(person_height), topL5, 
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)



cv2.imshow("Contours", canvas)
# cv2.imwrite(os.path.join(dirname, 'filtered_images/result_frontPt2.jpg'), canvas)

if cv2.waitKey(0) & 0xFF == ord('q'): 
    cv2.destroyAllWindows()