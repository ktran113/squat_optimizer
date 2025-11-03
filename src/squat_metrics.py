import numpy as np

COCO = dict(L_hip=11, R_hip=12, L_knee=13, R_knee=14,L_ank=15, R_ank=16)

def sideSelector(xy, con):
    # Figures out which side to select by taking the mean of keypoints on the 
    # left and the right sides and then chooses based on the higher mean.

    left = [COCO["L_hip"], COCO["L_knee"], COCO["L_ank"]]
    right = [COCO["R_hip"], COCO["R_knee"], COCO["R_ank"]]
    
    left_conf = np.mean(con[:, left])
    right_conf = np.mean(con[:, right])

    side = "L" if left_conf > right_conf else "R"
    print(f"Selected {side} for the side")

    hip = COCO[f"{side}_hip"]
    knee = COCO[f"{side}_knee"]
    ank = COCO[f"{side}_ank"]

    return(xy[:, hip, :], xy[:, knee, :], xy[:, ank, :])




    #A: Hip
    #B: Knee
    #C: Ankles

def angle(A, B, C): #cos(theta) = (v1 dot v2) / mag(v1) * mag(v2)
    v1 = A - B
    v2 = C - B

    dot = np.sum(v1 * v2, axis=1)
    mag_v1 = np.linalg.norm(v1, axis=1)
    mag_v2 = np.linalg.norm(v2, axis=1)

    cos = dot / (mag_v1 * mag_v2 + 1e-6)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))

def knee_angle(knee, hip, ank):
    return angle(hip, knee, ank)

def squat_depth(knee, hip): 
    knee = knee[:,1]
    hip = hip[:,1]

    depth = hip - knee

    
