# find coordinates of contours detected 
import numpy as np 
import cv2
import os

dirname = os.path.dirname(__file__)

# Reading image
font = cv2.FONT_HERSHEY_COMPLEX 
img2 = cv2.imread(os.path.join(dirname, '../input/test.jpg'), cv2.IMREAD_COLOR) 

# Same image in another variable and converting to gray scale. 
img = cv2.imread(os.path.join(dirname, '../input/test.jpg'), cv2.IMREAD_GRAYSCALE) 

# Converting image to a binary image 
# ( black and white only image). 
_, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY) 

# Detecting contours in image. 
contours, _= cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 

# Going through every contours found in the image. 
for cnt in contours:
    
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True) 
    print(type(approx))
    # draws boundary of contours. 
    cv2.drawContours(img2, [approx], -1, (0, 0, 255), 2) 

    # Used to flatted the array containing 
    # the co-ordinates of the vertices.
    n = approx.ravel()
    i = 0

    for j in n : 
        if(i % 2 == 0): 
            x = n[i] 
            y = n[i + 1] 

            # String containing the co-ordinates. 
            string = str(x) + " " + str(y) 

            if(i == 0): 
                # text on topmost co-ordinate. 
                cv2.putText(img2, "Arrow tip", (x, y), font, 0.5, (255, 255, 0)) 
            else: 
                # text on remaining co-ordinates. 
                cv2.putText(img2, string, (x, y), font, 0.5, (0, 255, 0)) 
        i = i + 1

# Showing the final image. 
cv2.imshow('image2', img2) 

# Exiting the window if 'q' is pressed on the keyboard. 
if cv2.waitKey(0) & 0xFF == ord('q'): 
    cv2.destroyAllWindows()