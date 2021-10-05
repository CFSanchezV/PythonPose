import numpy as np
from numpy.core.fromnumeric import shape

def calculate_Neck(a, b, w, h):
    a = np.array(a) # First
    b = np.array(b) # Mid

    a2 = np.multiply(a, [w, h]).astype(int)
    b2 = np.multiply(b, [w, h]).astype(int)
        
    dist = b2.copy()
    dist[1] -= a2[1]
    dist[1] = abs(dist[1])

    return dist


def calculate_Waist(ls, rs, lh, rh,  w, h):
    ls = np.array(ls) # l should
    lh = np.array(lh) # l hip

    rs = np.array(rs) # r should
    rh = np.array(rh) # r hip

    la = np.multiply(ls, [w, h]).astype(int)
    lb = np.multiply(lh, [w, h]).astype(int)
    ra = np.multiply(rs, [w, h]).astype(int)
    rb = np.multiply(rh, [w, h]).astype(int)
    
    lw = np.copy(la)
    lw[0] = abs(la[0] - lb[0])
    lw[1] = abs(lb[1] - la[1])

    rw = np.copy(ra)
    rw[0] = abs(rb[0] - ra[0])
    rw[1] = abs(rb[1] - ra[1])

    print("lw:", lw)
    print("rw:", rw)

    return lw, rw




# w, h = 600, 400
# ls= [3,3]
# rs= [13,3]
# lh= [6,11]
# rh= [9,11]

# # y pos must be p2.y - p1.y
# # xypos = calculate_Waist(ls, rs, lh, rh, w, h)

# x = np.ones((5, 5), np.float32)/1
# print(x)





'''
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
'''


class PositionsSide():

    def __init__(self, *args):
        print(len(args))
        if len(args) == 4:
            self.LneckY = args[0]
            self.RneckY = args[1]
            self.chestY = args[2]
        else:
            print("mal")


shap = PositionsSide (2, 34, 5, 76)
print(shap.RneckY)