import cv2
import mediapipe as mp
import time
import os
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

dirname = os.path.dirname(__file__)


def calc_waist():
    pass

# For static images:
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:

    image = cv2.imread(os.path.join(dirname, 'input/fashion1.jpg'))  #Insert your Image Here
    image_height, image_width, _  = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark
    except:
        pass


    # Draw pose landmarks on the image.
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow("annotated", annotated_image)
    
    cv2.waitKey()
    if cv2.waitKey(10) & 0xFF == ord('q'): quit()
    #cv2.imwrite(r'anotada.png', annotated_image)


#GET landmarks necessary
#landmarks positions for needed values: [11 to 16] for arms && [23 to 30] for arms
l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]


# Get contuor image if white bg




# Get conversion factor
pixelsPerMetric = None

#example measurements in cm
chest_circumf = None
waist_circumf = None
hip_circumf = None


def pixeldensity():
    pass