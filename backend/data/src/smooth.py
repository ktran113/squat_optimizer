from scipy.signal import savgol_filter
import numpy as np

SMOOTH_SECONDS = 0.3    #time window to smooth over, converted to frames using fps
POLY_ORDER = 2
DEFAULT_FPS = 30

def window_frames(fps):
    """
    Converts the smoothing window from seconds to frames. savgol needs an odd
    window longer than the polynomial order.
    """
    frames = int(round(SMOOTH_SECONDS * (fps or DEFAULT_FPS)))
    if frames % 2 == 0:
        frames += 1
    return max(frames, POLY_ORDER + 3)

def smooth(xy, conf_valid, fps=DEFAULT_FPS):
    time = np.arange(len(xy))
    length = xy.shape[0]
    output = np.asarray(xy.copy(), dtype=np.float32)
    window = window_frames(fps)

    if length < window:     #Video too short to smooth out
        return xy

    for direction in [0,1]:
        values = output[:, direction].copy()

        #Stores frames of bad / good data
        good = conf_valid & np.isfinite(values)
        bad = ~good
        if np.sum(good) < 2:
            return xy
        values[bad] = np.interp(time[bad], time[good], values[good])
        output[:, direction] = savgol_filter(values, window, POLY_ORDER)

    return output
