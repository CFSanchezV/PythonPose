import cv2
import numpy as np
import os
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

dirname = os.path.dirname(__file__)
BLACK = (0, 0, 0)


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


def find_FootY(la, w, h, img):
    a = np.array(la) # l heel
    lh = np.multiply(a, [w, h]).astype(int)

    # cv2.circle(img, (w//2, lh[1]), 5, BLACK, 2)

    return lh[1]


def find_NeckY(a, b, w, h, img):
    a = np.array(a) # nose
    b = np.array(b) # should

    a2 = np.multiply(a, [w, h]).astype(int)
    b2 = np.multiply(b, [w, h]).astype(int)

    p = b2.copy() # should
    ydiff = abs(a2[1] - b2[1])
    y2 = (ydiff)/2
    p[1] = b2[1] - y2

    # cv2.circle(img, (w//2, p[1]), 5, BLACK, 2)

    return p[1]


def find_ChestY(a, b, w, h, img):
    a = np.array(a) # elbow
    b = np.array(b) # should

    a2 = np.multiply(a, [w, h]).astype(int)
    b2 = np.multiply(b, [w, h]).astype(int)
        
    p = a2.copy() # elbow
    ydiff = abs(a2[1] - b2[1])
    y2 = (ydiff)/2
    p[1] = b2[1] + y2

    # cv2.circle(img, (w//2, p[1]), 5, BLACK, 2)

    return p[1]


def calculate_Chest(lel, lsh, rel, rsh, w, h, img):
    a = np.array(lsh) # l should
    b = np.array(rsh) # r should

    ls = np.multiply(a, [w, h]).astype(int) # l should
    rs = np.multiply(b, [w, h]).astype(int) # r should

    LY = find_ChestY(lel, lsh, w, h, img)
    RY = find_ChestY(rel, rsh, w, h, img)

    ls[1] = LY
    rs[1] = RY

    cv2.circle(img, ls, 5, BLACK, 2)
    cv2.circle(img, rs, 5, BLACK, 2)

    dist = np.linalg.norm(ls - rs)

    cv2.line(img, ls, rs, BLACK, 2, cv2.LINE_AA)

    return dist


def calculate_Waist(la, ra, lb, rb,  w, h, img):
    a = np.array(la) # l should
    b = np.array(ra) # r should
    c = np.array(lb) # l hip
    d = np.array(rb) # r hip

    ls = np.multiply(a, [w, h]).astype(int) # l should
    rs = np.multiply(b, [w, h]).astype(int) # r should
    lh = np.multiply(c, [w, h]).astype(int) # l hip
    rh = np.multiply(d, [w, h]).astype(int) # r hip

    lw = np.copy(ls)
    lw[0] = (abs(ls[0] - lh[0])) /2 + lh[0]
    lw[1] = (abs(lh[1] - ls[1])) /2 + ls[1]

    rw = np.copy(rs)
    rw[0] = (abs(rh[0] - rs[0])) /2 + rs[0]
    rw[1] = (abs(rh[1] - rs[1])) /2 + rs[1]

    # cv2.circle(img, lw, 5, BLACK, 2)
    # cv2.circle(img, rw, 5, BLACK, 2)

    dist = np.linalg.norm(lw - rw)

    # cv2.line(img, rw, lw, BLACK, 2, cv2.LINE_AA)

    return dist


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
    lw[0] = (abs(ls[0] - lh[0])) /2 + lh[0]
    lw[1] = (abs(lh[1] - ls[1])) /2 + ls[1]

    rw = np.copy(rs)
    rw[0] = (abs(rh[0] - rs[0])) /2 + rs[0]
    rw[1] = (abs(rh[1] - rs[1])) /2 + rs[1]

    cv2.circle(img, lw, 5, BLACK, 2)
    cv2.circle(img, rw, 5, BLACK, 2)

    # dist = np.linalg.norm(lw - rw)
    # cv2.line(img, rw, lw, BLACK, 2, cv2.LINE_AA)

    return lw[1], rw[1]


def find_HipY(a, w, h, img):
    ar = np.array(a) # First
    a2 = np.multiply(ar, [w, h]).astype(int)

    # cv2.circle(img, (w//2, a2[1]), 5, BLACK, 2)

    return a2[1]


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


with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    ### FRONT ###
    img_front = cv2.imread(os.path.join(dirname, 'filtered_images/result_front.jpg'))
    img_front = image_resize(img_front, height=730)
    h, w, _  = img_front.shape
    print("F Pos height:", h)
    print("F Pos width:", w)
    
    # Recolor image to RGB
    img_front = cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB)

    # Make detections
    results = pose.process(img_front)
    
    # Recolor back to BGR
    img_front = cv2.cvtColor(img_front, cv2.COLOR_RGB2BGR)

    annotated_front = img_front.copy()
    
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

    # Find positions
    topL1, topL2, topL3, topL4, topL5 = (0,15), (0,30), (0,45), (0,60), (0,75)
    neckY = find_NeckY(nose, l_shoulder, w, h, annotated_front)
    chest_dist_front = calculate_Chest(l_elbow, l_shoulder, r_elbow, r_shoulder, w, h, annotated_front)
    LwaistY, RwaistY = find_WaistY(l_shoulder, r_shoulder, l_hip, r_hip,  w, h, annotated_front)
    hipY = find_HipY(l_hip, w, h, annotated_front)
    footY = find_FootY(l_heel, w, h, annotated_front)

    # Calculate leg/arm sizes
    l_arm = calculate_Dist(l_shoulder, l_elbow, l_wrist, w, h)
    r_arm = calculate_Dist(r_shoulder, r_elbow, r_wrist, w, h)

    l_leg = calculate_Dist(l_hip, l_knee, l_ankle, w, h, d=l_heel)
    r_leg = calculate_Dist(r_hip, r_knee, r_ankle, w, h, d=r_heel)

    # Visualize y_positions, and additional points
    cv2.putText(annotated_front, "neck(_,y): {}".format(neckY), topL1, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
    
    cv2.putText(annotated_front, "chest front: {:.3f}".format(chest_dist_front), topL2, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

    cv2.putText(annotated_front, "waist(left, right): {}, {}".format(LwaistY, RwaistY), topL3, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

    cv2.putText(annotated_front, "hip(_,y): {}".format(hipY), topL4, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

    cv2.putText(annotated_front, "foot(_,y): {}".format(footY), topL5, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)


    # Visualize leg/arm sizes
    cv2.putText(annotated_front, "{:.3f}".format(l_arm), 
                    tuple(np.multiply(l_elbow, [w, h]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

    cv2.putText(annotated_front, "{:.3f}".format(r_arm), 
                    tuple(np.multiply(r_elbow, [w, h]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

    cv2.putText(annotated_front, "{:.3f}".format(l_leg), 
                    tuple(np.multiply(l_knee, [w, h]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

    cv2.putText(annotated_front, "{:.3f}".format(r_leg), 
                    tuple(np.multiply(r_knee, [w, h]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
 
    
    mp_drawing.draw_landmarks(annotated_front, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) )


cv2.imshow('Imagen anotada', annotated_front)
    
if cv2.waitKey(0) & 0xFF == ord('q'): 
    cv2.destroyAllWindows()