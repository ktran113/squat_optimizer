import numpy as np
from scipy.signal import find_peaks

#Keypoint indices for COCO pose format
COCO = dict(L_hip=11, R_hip=12, L_knee=13, R_knee=14,L_ank=15, R_ank=16)

EPSILON = 1e-6  #small value to prevent division by zero
MIN_DEPTH_THRESHOLD = 0  #min depth threshold for rep detection
MIN_DISTANCE_BETWEEN_REPS = 30  #min frames between rep peaks
ANGLE_BELOW_PARALLEL = 90  #knee angle threshold for "below parallel" (degrees)
ANGLE_PARALLEL = 100  #knee angle threshold for "parallel" (degrees)
DEFAULT_REP_WINDOW = 15  #default frame window around rep peak
MIN_FRAMES_BAR_PATH = 15  # minimum frames needed for bar path analysis
HIP_HEEL_ERROR_THRESHOLD = 50  #pixel threshold for hip-heel alignment

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

    cos = dot / (mag_v1 * mag_v2 + EPSILON)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))

def knee_angle(hip, knee, ank):
    return angle(hip, knee, ank)

def squat_depths (hip, knee):
    """
    Returns squat depths, knee - hip because with OpenCV 
    lower coordinates mean higher value
    """
    knee = knee[:,1]
    hip = hip[:,1]
    return knee - hip 

def hip_heel(hip, ank, error):
    """
    Checks whether or not hips and ankles are lined up vertically
    Returning a numpy list of true or false regarding if user hit depth
    """
    lineup = hip[:,0] - ank[:,0]
    result = np.where(np.abs(lineup) <= error, True, False)        
    return result

def rep_count(hip, knee): 
    """
    Returns numpy array of bottom frames
    Length of array gives you # of reps
    """
    depth_over_time = squat_depths(hip, knee)
    peaks, _ = find_peaks(depth_over_time, height=MIN_DEPTH_THRESHOLD, distance=MIN_DISTANCE_BETWEEN_REPS)
    return peaks
    #depth_over_time[peaks] >= error

def rep_tempo(rep_bottom_frames, fps = 30):
    """
    Returns an array of how long each rep took
    """
    rep_times = np.diff(rep_bottom_frames) / fps
    return rep_times

def depth_quality(hip, knee, ank, peaks):
    """
    Analyzes angles of the bottoms of each rep 
    Returning an array where each index correlates to a rep
    containing whether or not the lifter completed the rep 
    below, parallel, or paritally.
    """
    bottom_angles = knee_angle(hip, knee, ank)[peaks]
    quality = []
    for angle in bottom_angles:
        if angle < ANGLE_BELOW_PARALLEL:
            quality.append("below")
        elif angle < ANGLE_PARALLEL:
            quality.append("parallel")
        else:
            quality.append("partial")
    return quality

def segment_reps(hip, knee, ank, peaks, window=DEFAULT_REP_WINDOW):
    """
    Returns a dictionary containing information per rep, 
    information contains windows, rep count, bottom frame
    depth, and the angle at the bottom
    """
    reps = []
    depth = depth_quality(hip, knee, ank, peaks)
    for i, peak in enumerate(peaks):
        start = max(0, peak - window)
        end = min(len(hip), peak + window)

        rep_stats = {
            "rep_count": i + 1,
            "bottom_frame": peak, 
            "start": start,
            "end" : end, 
            "depth" : depth[i],
            "bottom_angle" : angle(hip[peak:peak+1], knee[peak: peak +1 ], ank[peak: peak + 1])[0]
        }
        reps.append(rep_stats)
    return reps

def bar_path_analysis(barbell_xy, start, end):
    """
    Analyzes barpath returning the deviation of the barbells
    horizontal movement during reps excluding the top of the 
    rep where the lifter might reset and adjust the bar
    """
    window = barbell_xy[start:end]
    bar_x = window[:, 0]
    bar_x = bar_x[~np.isnan(bar_x)]

    if len(bar_x) < MIN_FRAMES_BAR_PATH:
        return np.nan

    horizontal_dev = np.std(bar_x)
    return horizontal_dev

def analyze_squat(xy, conf, barbell_xy, fps=30):
    """
    Returns a dictionary with the results of complete analysis 
    on the squat video
    """
    hip, knee, ank = sideSelector(xy, conf)
    peaks = rep_count(hip, knee)
    reps = segment_reps(hip, knee, ank, peaks)
    knee_ang = knee_angle(hip, knee, ank)
    depth = squat_depths(hip,knee)
    hip_heel_alignment = hip_heel(hip, ank, error=HIP_HEEL_ERROR_THRESHOLD)
    tempo = rep_tempo(peaks, fps)
    bar_dev = []

    for rep in reps:
        bar_dev.append(bar_path_analysis(barbell_xy, rep["start"], rep["end"]))
    return {
        "total_reps": len(peaks),   # Returns # of total reps
        "reps": reps,               #Segmented reps dict
        "depth_over_time": depth,   #Returns array of ank - hip y coords (lower on screen higher value)
        "knee_angle": knee_ang,     #Returns knee angle over entire time series
        "tempo_per_rep": tempo,     #Returns array of tempo of that rep
        "hip_heel_alignment": hip_heel_alignment,   #Returns time series of hip_heel alignment
        "bar_path_dev": bar_dev     #Returns time series of deviations of the bar path per rep (float)
    }

    
