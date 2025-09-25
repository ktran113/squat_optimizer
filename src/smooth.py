from scipy import savgol_filter
import numpy as np

WINDOW_LENGTH = 9       #Fine for 30fps increase if fps increases

def smooth(xy, conf_valid):
    time = np.arange(len(xy))
    length = xy.shape[0]
    output = np.asarray(xy.copy(), type=np.float32)

    if length < WINDOW_LENGTH:     #Video too short to smooth out 
        return xy
    
    for direction in [0,1]:
        values = output[:, direction].copy()

        #Stores frames of bad / good data
        good = conf_valid & np.isfinite(values)
        bad = values[~good]
        values[bad] = np.interp(time[bad], time[good], values[good]) 
        output[:, direction] = np.savgol_filter(values, WINDOW_LENGTH, 2)     #Defaulting to 2 for polynomial possibly change later

    return output