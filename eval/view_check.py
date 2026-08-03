"""
Heuristics for deciding whether a video is usable by the squat pipeline.

Two independent checks:

  view_ratio   how frontal the camera is. Left/right keypoints project onto
               nearly the same x from the side and separate as the camera moves
               to the front, so the left-right x gap over torso length is near 0
               for a side view and grows toward a frontal one. Dividing by torso
               length keeps it independent of resolution and camera distance.

  pose_conf    whether pose estimation worked at all. Takes the better-detected
               side per frame, so a normally occluded far leg does not count
               against an otherwise clean video.

Thresholds are intentionally not baked in. Run run_eval.py --calibrate over
videos whose view you already know, then pass the cutoff you pick.
"""

import numpy as np

L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
LEFT_LEG = [11, 13, 15]
RIGHT_LEG = [12, 14, 16]

MIN_CONF = 0.3          # keypoint confidence needed to use a frame for geometry
MIN_TORSO_PX = 1e-6


def view_ratio(xy, conf):
    """
    Median left-right horizontal separation over torso length.
    ~0.0-0.1 side view, higher as the camera moves frontal. NaN if unmeasurable.
    """
    pairs = [(L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP)]
    usable = np.ones(len(xy), dtype=bool)
    for a, b in pairs:
        usable &= (conf[:, a] > MIN_CONF) & (conf[:, b] > MIN_CONF)
    usable &= np.all(np.isfinite(xy[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP], :]), axis=(1, 2))

    if not np.any(usable):
        return np.nan

    frames = xy[usable]
    widths = np.maximum(
        np.abs(frames[:, L_SHOULDER, 0] - frames[:, R_SHOULDER, 0]),
        np.abs(frames[:, L_HIP, 0] - frames[:, R_HIP, 0]),
    )
    shoulder_y = (frames[:, L_SHOULDER, 1] + frames[:, R_SHOULDER, 1]) / 2
    hip_y = (frames[:, L_HIP, 1] + frames[:, R_HIP, 1]) / 2
    torso = np.abs(shoulder_y - hip_y)

    ok = torso > MIN_TORSO_PX
    if not np.any(ok):
        return np.nan
    return float(np.median(widths[ok] / torso[ok]))


def pose_conf(conf):
    """
    Mean confidence of the better-detected leg, averaged over frames.
    Low means pose estimation failed, not that the subject is side-on.
    """
    left = np.mean(conf[:, LEFT_LEG], axis=1)
    right = np.mean(conf[:, RIGHT_LEG], axis=1)
    return float(np.mean(np.maximum(left, right)))


def check(xy, conf, max_view_ratio=None, min_pose_conf=None):
    """
    Returns (ok, ratio, quality, reason). ok is True when no threshold was
    given, so filtering is always opt-in rather than silent.
    """
    ratio = view_ratio(xy, conf)
    quality = pose_conf(conf)

    if min_pose_conf is not None and quality < min_pose_conf:
        return False, ratio, quality, f"pose_conf {quality:.2f} < {min_pose_conf}"
    if max_view_ratio is not None and np.isfinite(ratio) and ratio > max_view_ratio:
        return False, ratio, quality, f"view_ratio {ratio:.2f} > {max_view_ratio}"
    return True, ratio, quality, ""
