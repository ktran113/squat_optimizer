import numpy as np
from scipy.signal import find_peaks

COCO = dict(L_hip=11, R_hip=12, L_knee=13, R_knee=14,L_ank=15, R_ank=16)

def sideSelector(xy, con):
    """
    Selects side to look at by evaluating higher average confidence
    """

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

    #A Hip
    #B Knee
    #C Ankles
def angle(A, B, C): #cos(theta) = (v1 dot v2) / mag(v1) * mag(v2)
    v1 = A - B
    v2 = C - B

    dot = np.sum(v1 * v2, axis=1)
    mag_v1 = np.linalg.norm(v1, axis=1)
    mag_v2 = np.linalg.norm(v2, axis=1)

    cos = dot / (mag_v1 * mag_v2 + 1e-6)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))

def knee_angle(hip, knee, ank):
    return angle(hip, knee, ank)

def squat_depths (hip, knee):
    knee = knee[:,1]
    hip = hip[:,1]
    return hip - knee

def hip_heel(hip, ank, error):
    lineup = hip[:,0] - ank[:,0]
    result = []
    for i in range(len(lineup)):
        if lineup[i] > abs(lineup[i] + error):
            result.append(0)
        else:
            result.append(1)

def rep_count(hip, knee): 
    """
    Each peak in the array of depths increase the rep count
    """
    #Values below are error bounds for depth percision
    MIN_DEPTH_THRESHHOLD = 0
    MIN_DISTANCE = 30

    depth_over_time = squat_depths(hip, knee)
    peaks, _ = find_peaks(depth_over_time, MIN_DEPTH_THRESHHOLD, MIN_DISTANCE)
    return peaks
    #depth_over_time[peaks] >= error

def rep_tempo(rep_bottom_frames, fps = 30):
    """
    Returns an array of how long each rep took
    """
    rep_times = np.diff(rep_bottom_frames) / fps
    return rep_times


    
