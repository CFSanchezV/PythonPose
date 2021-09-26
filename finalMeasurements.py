import cv2
import mediapipe as mp
import time
import os
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

dirname = os.path.dirname(__file__)
font = cv2.FONT_HERSHEY_COMPLEX 

# Simulate receiving countour processed image
fore_image = cv2.imread(os.path.join(dirname, 'output/justforeground.jpg'), cv2.IMREAD_COLOR) 
cv2.imshow('Imagen inicial', fore_image)

image = cv2.imread(os.path.join(dirname, 'output/justforeground.jpg'), cv2.IMREAD_GRAYSCALE)

# black and white only image
_, threshold = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)

# Get contuor image if white bg
contours, _= cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Going through every contour
for cnt in contours:
    
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
    # draws boundary of contours. 
    cv2.drawContours(image, [approx], -1, (144, 238, 144), 2)

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
                cv2.putText(image, "Arrow tip", (x, y), font, 0.5, (0, 255, 0)) 
            else: 
                # text on remaining co-ordinates. 
                cv2.putText(image, string, (x, y), font, 0.5, (0, 255, 0)) 
        i = i + 1

# Showing the final image. 
cv2.imshow('Contours', image)

if cv2.waitKey(0) & 0xFF == ord('q'): 
    cv2.destroyAllWindows()


# Find positions for final measurements
def calculate_Neck():
    pass

def calculate_Chest():
    pass

def calculate_Hip():
    pass