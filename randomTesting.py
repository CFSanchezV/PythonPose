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




w, h = 600, 400
ls= [3,3]
rs= [13,3]
lh= [6,11]
rh= [9,11]

# y pos must be p2.y - p1.y
# xypos = calculate_Waist(ls, rs, lh, rh, w, h)

x = np.ones((5, 5), np.float32)/1
print(x)
