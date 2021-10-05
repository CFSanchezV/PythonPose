import cv2
import mediapipe as mp
import numpy as np
import imutils
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import os

# VIDEO FEED
"""
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('Mediapipe Feed', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
"""

global counter
counter = 0
dirname = os.path.dirname(__file__)
picpath = os.path.join(dirname, 'picturesTaken')

def takePictures(ret, frame, picpath):
    if not ret:
        print("failed to grab frame")
    k = cv2.waitKey(1)
    if k%256 == 32:
        # SPACE key press to take pic
        global counter
        img_name = "opencv_frame_{}.png".format(counter)
        cv2.imwrite(os.path.join(picpath , img_name), frame)
        print("{} written!".format(img_name))
        counter += 1


cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Resizing | initial dimen: (640, 480)
        frame = imutils.resize(frame, width=840)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 630)
        w = 840
        h = 630
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)
        #takePictures(ret, image, picpath)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        else:
            takePictures(ret, image, picpath)

    cap.release()
    


img_name = "opencv_frame_{}.png".format(counter)
lastImg = cv2.imread(os.path.join(picpath , img_name))

cv2.imshow("img", lastImg)
cv2.destroyAllWindows()
