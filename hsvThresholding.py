import cv2
import os
import numpy as np

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


dirname = os.path.dirname(__file__)


path = os.path.join(dirname, "input/person.jpg")
img = cv2.imread(path)

#resizer for test
img = image_resize(img, height=600)


imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


#trackbars
cv2.namedWindow("trackbar")
cv2.resizeWindow("trackbar", 640, 240)

def empty():
    pass
cv2.createTrackbar("h min", "trackbar" , 0, 179, empty)
cv2.createTrackbar("h max", "trackbar" , 179, 255, empty)
cv2.createTrackbar("s min", "trackbar" , 0, 255, empty)
cv2.createTrackbar("s max", "trackbar" , 255, 255, empty)
cv2.createTrackbar("v min", "trackbar" , 0, 255, empty)
cv2.createTrackbar("v max", "trackbar" , 255, 255, empty)

'''
while True:
    img = image_resize(img, height=600)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("h min", "trackbar")
    h_max = cv2.getTrackbarPos("h max", "trackbar")
    s_min = cv2.getTrackbarPos("s min", "trackbar")
    s_max = cv2.getTrackbarPos("s max", "trackbar")
    v_min = cv2.getTrackbarPos("v min", "trackbar")
    v_max = cv2.getTrackbarPos("v max", "trackbar")
    lower = np.array([0, 50, 0])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(imgHSV, lower, upper)


    cv2.imshow("Original", img)
    cv2.imshow("mask", mask)
    cv2.imshow("HSV", imgHSV)

    cv2.waitKey(1)
'''  

lower = np.array([0, 45, 0])
upper = np.array([255, 255, 255])

mask = cv2.inRange(imgHSV, lower, upper)

imgResult = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("Original", img)
cv2.imshow("mask", mask)
cv2.imshow("Result", imgResult)
cv2.waitKey()
cv2.destroyAllWindows()