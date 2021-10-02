import cv2
import mediapipe as mp
import numpy as np
import os
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

dirname = os.path.dirname(__file__)

def calculate_Dist(a, b, c, w, h, d=None):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    dist = None

    if d is None:
        a2 = np.multiply(a, [w, h]).astype(int)
        b2 = np.multiply(b, [w, h]).astype(int)
        c2 = np.multiply(c, [w, h]).astype(int)
        
        dist1 = np.linalg.norm(a2 - b2)
        dist2 = np.linalg.norm(b2 - c2)
        dist = abs(dist1) + abs(dist2)
    else:
        d = np.array(d)
        a2 = np.multiply(a, [w, h]).astype(int)
        b2 = np.multiply(b, [w, h]).astype(int)
        c2 = np.multiply(c, [w, h]).astype(int)
        d2 = np.multiply(c, [w, h]).astype(int)

        dist1 = np.linalg.norm(a2 - b2)
        dist2 = np.linalg.norm(b2 - c2)
        dist3 = np.linalg.norm(c2 - d2)
        dist = abs(dist1) + abs(dist2)+ abs(dist3)

    return dist


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

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
landmarks = None
# Mediapipe instance
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    image = cv2.imread(os.path.join(dirname, 'filtered_images/result_front.jpg'))
    if image is None:
        print("img not found!!!")
        quit()

    image = image_resize(image, height=730)
    h, w, _  = image.shape

    # Recolor image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Make detections
    results = pose.process(image)
    
    # Recolor back to BGR
    img_front = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    annotated_image = image.copy()

    # Extract landmarks
    landmarks = results.pose_landmarks.landmark
    #landmarks positions for needed values: [11 to 16] for arms && [23 to 30] for arms
    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    
    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    l_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
    r_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]

    # Calculate leg/arm sizes
    l_arm = calculate_Dist(l_shoulder, l_elbow, l_wrist, w, h)
    r_arm = calculate_Dist(r_shoulder, r_elbow, r_wrist, w, h)

    l_leg = calculate_Dist(l_hip, l_knee, l_ankle, w, h, d=l_heel)
    r_leg = calculate_Dist(r_hip, r_knee, r_ankle, w, h, d=r_heel)

    # print("{:.2f}".format(l_arm)) #rounded 3f == str(l_arm)
    # Visualize sizes
    # print(l_arm)
    cv2.putText(annotated_image, "{:.3f}".format(l_arm), 
                    tuple(np.multiply(l_elbow, [w, h]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

    cv2.putText(annotated_image, "{:.3f}".format(r_arm), 
                    tuple(np.multiply(r_elbow, [w, h]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

    cv2.putText(annotated_image, "{:.3f}".format(l_leg), 
                    tuple(np.multiply(l_knee, [w, h]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

    cv2.putText(annotated_image, "{:.3f}".format(r_leg), 
                    tuple(np.multiply(r_knee, [w, h]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
    
    # Render detections
    mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) )
                                
    cv2.imshow('Imagen anotada', annotated_image)
    
    if cv2.waitKey(0) & 0xFF == ord('q'): 
        cv2.destroyAllWindows()
        

print("")
print("")