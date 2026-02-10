from ultralytics import YOLO
import numpy as np

def run_detection(path, weights_path):
    """
    runs barbell detection
    returns (xy, conf), shape (n, 2) and (n,)
    """
    model = YOLO(weights_path)

    print("detecting barbell locally")
    xy = []
    conf = []

    results = model(path, stream=True, verbose=False)
    for frame_result in results:
        if frame_result.boxes is not None and len(frame_result.boxes) > 0:
            best_idx = int(frame_result.boxes.conf.argmax().item())
            box = frame_result.boxes.xywh[best_idx].cpu().numpy()
            c = float(frame_result.boxes.conf[best_idx].cpu().numpy())
            xy.append([box[0], box[1]])
            conf.append(c)
        else:
            xy.append([np.nan, np.nan])
            conf.append(0.0)

    print(f"Done barbell detection {len(xy)} frames processed")
    return np.array(xy, dtype=np.float32), np.array(conf, dtype=np.float32)