import math
import cv2
import mediapipe as mp
import numpy as np
import os
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

dirname = os.path.dirname(__file__)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def calculate_Perimeter(a, b):
    perimeter = math.pi * ( 3*(a + b) - math.sqrt( (3*a + b) * (a + 3*b) ))
    return perimeter


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


def find_NeckY(a, b, c, w, h, img):
    a = np.array(a) # nose
    b = np.array(b) # l should
    c = np.array(c) # mouth r

    d_mouth_nose = abs(c[1] - a[1])

    n = np.multiply(a, [w, h]).astype(int)
    ls = np.multiply(b, [w, h]).astype(int)
    mr = np.multiply(c, [w, h]).astype(int)

    # l neck point
    lneck = ls.copy() # l neck
    ydiff = abs(n[1] - ls[1])
    y2 = (ydiff)//2
    lneck[1] = ls[1] - y2

    # r neck point
    d_mouth_nose = abs(mr[1] - n[1])

    rneck = lneck.copy()
    rneck[1] += d_mouth_nose

    cv2.circle(img, (230, lneck[1]), 5, WHITE, 2)
    cv2.circle(img, (270, rneck[1]), 5, WHITE, 2)

    return lneck[1], rneck[1]


def find_ChestY(a, b, w, h, img):
    a = np.array(a) # elbow
    b = np.array(b) # should

    a2 = np.multiply(a, [w, h]).astype(int)
    b2 = np.multiply(b, [w, h]).astype(int)
        
    p = a2.copy() # elbow
    ydiff = abs(a2[1] - b2[1])
    y2 = (ydiff)/2
    p[1] = b2[1] + y2

    cv2.circle(img, (240, p[1]), 5, BLACK, 2)

    return p[1]


def find_WaistY(la, ra, lb, rb,  w, h, img):
    a = np.array(la) # l should
    b = np.array(ra) # r should
    c = np.array(lb) # l hip
    d = np.array(rb) # r hip

    ls = np.multiply(a, [w, h]).astype(int) # l should
    rs = np.multiply(b, [w, h]).astype(int) # r should
    lh = np.multiply(c, [w, h]).astype(int) # l hip
    rh = np.multiply(d, [w, h]).astype(int) # r hip

    lw = np.copy(ls)
    lw[1] = (abs(lh[1] - ls[1])) /2 + ls[1]

    rw = np.copy(rs)
    rw[1] = (abs(rh[1] - rs[1])) /2 + rs[1]

    p = lw.copy() # l waist
    cv2.circle(img, (240, p[1]), 5, BLACK, 2)

    return p[1]


def find_HipY(a, w, h, img):
    ar = np.array(a) # First
    a2 = np.multiply(ar, [w, h]).astype(int)

    cv2.circle(img, (240, a2[1]), 5, BLACK, 2)

    return a2[1]


landmarks = None
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    ### SIDE ###
    img_side = cv2.imread(os.path.join(dirname, 'filtered_images/result_side.jpg'))
    img_side = image_resize(img_side, height=730)
    h, w, _  = img_side.shape

    # Recolor image to RGB
    img_side = cv2.cvtColor(img_side, cv2.COLOR_BGR2RGB)

    # Make detections
    results = pose.process(img_side)
    
    # Recolor back to BGR
    img_side = cv2.cvtColor(img_side, cv2.COLOR_RGB2BGR)

    annotated_side = img_side.copy()
    
    # GET LANDMARKS
    landmarks = results.pose_landmarks.landmark
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

    nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
    mouth_r = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x,landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]

    # Find positions
    topL1, topL2, topL3, topL4 = (0,15), (0,30), (0,45), (0,60)
    LneckY, RneckY = find_NeckY(nose, l_shoulder, mouth_r, w, h, annotated_side)
    chestY = find_ChestY(l_elbow, l_shoulder, w, h, annotated_side)
    waistY = find_WaistY(l_shoulder, r_shoulder, l_hip, r_hip,  w, h, annotated_side)
    hipY = find_HipY(l_hip, w, h, annotated_side)

    # Visualize y_positions, additional points
    cv2.putText(annotated_side, "neck(left, right): {:.3f},{:.3f}".format(LneckY, RneckY), topL1, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
    
    cv2.putText(annotated_side, "chest(_, y): {:.3f}".format(chestY), topL2, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

    cv2.putText(annotated_side, "waist(_, y): {:.3f}".format(waistY), topL3, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

    cv2.putText(annotated_side, "hip(_, y): {:.3f}".format(hipY), topL4, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
 
    
    mp_drawing.draw_landmarks(annotated_side, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) )
                                
    cv2.imshow('Imagen anotada', annotated_side)
    
    if cv2.waitKey(0) & 0xFF == ord('q'): 
        cv2.destroyAllWindows()