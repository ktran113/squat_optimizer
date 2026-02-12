from ultralytics import YOLO
import numpy as np
import cv2

def run_detection(path, weights_path, skip=3):
    """
    runs barbell detection with frame skipping
    processes every `skip`th frame and interpolates the rest
    returns (xy, conf), shape (total_frames, 2) and (total_frames,)
    """
    model = YOLO(weights_path)

    # count total frames
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"detecting barbell locally ({total_frames} frames, skip={skip})")
    sampled_indices = []
    sampled_xy = []
    sampled_conf = []

    results = model(path, stream=True, verbose=False)
    for idx, frame_result in enumerate(results):
        if idx % skip != 0:
            continue
        sampled_indices.append(idx)
        if frame_result.boxes is not None and len(frame_result.boxes) > 0:
            best_idx = int(frame_result.boxes.conf.argmax().item())
            box = frame_result.boxes.xywh[best_idx].cpu().numpy()
            c = float(frame_result.boxes.conf[best_idx].cpu().numpy())
            sampled_xy.append([box[0], box[1]])
            sampled_conf.append(c)
        else:
            sampled_xy.append([np.nan, np.nan])
            sampled_conf.append(0.0)

    sampled_indices = np.array(sampled_indices)
    sampled_xy = np.array(sampled_xy, dtype=np.float32)
    sampled_conf = np.array(sampled_conf, dtype=np.float32)

    # interpolate to fill all frames
    all_indices = np.arange(total_frames)
    xy = np.full((total_frames, 2), np.nan, dtype=np.float32)
    conf = np.zeros(total_frames, dtype=np.float32)

    for dim in range(2):
        valid = ~np.isnan(sampled_xy[:, dim])
        if np.sum(valid) >= 2:
            xy[:, dim] = np.interp(all_indices, sampled_indices[valid], sampled_xy[valid, dim])
    conf[:] = np.interp(all_indices, sampled_indices, sampled_conf)

    print(f"Done barbell detection {len(sampled_indices)} frames processed, {total_frames} interpolated")
    return xy, conf