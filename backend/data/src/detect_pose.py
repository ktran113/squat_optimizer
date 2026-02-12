#Extracts keypoints and confidence using yolov8 pose estimation models
#returns an array of coordinates their confidences
from ultralytics import YOLO
import numpy as np
import cv2

def run_pose(video_path, weights='yolov8s-pose.pt', skip=2):
    model = YOLO(weights)

    # count total frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Running pose estimation ({total_frames} frames, skip={skip})")
    sampled_indices = []
    sampled_xy = []
    sampled_con = []

    results = model(video_path, stream=True)
    for idx, frame in enumerate(results):
        if idx % skip != 0:
            continue
        sampled_indices.append(idx)

        if frame.keypoints is None or len(frame.keypoints) == 0:
            sampled_xy.append(np.full((17, 2), np.nan, dtype=np.float32))
            sampled_con.append(np.zeros((17,), dtype=np.float32))
            continue

        person_idx = int(frame.boxes.conf.argmax().item())
        sampled_xy.append(frame.keypoints.xy[person_idx].cpu().numpy().astype(np.float32))

        if frame.keypoints.conf is not None:
            sampled_con.append(frame.keypoints.conf[person_idx].cpu().numpy().astype(np.float32))
        else:
            sampled_con.append(np.zeros((17,), dtype=np.float32))

    sampled_indices = np.array(sampled_indices)
    sampled_xy = np.stack(sampled_xy)
    sampled_con = np.stack(sampled_con)

    # interpolate to fill all frames
    all_indices = np.arange(total_frames)
    xy = np.full((total_frames, 17, 2), np.nan, dtype=np.float32)
    con = np.zeros((total_frames, 17), dtype=np.float32)

    for joint in range(17):
        for dim in range(2):
            valid = ~np.isnan(sampled_xy[:, joint, dim])
            if np.sum(valid) >= 2:
                xy[:, joint, dim] = np.interp(all_indices, sampled_indices[valid], sampled_xy[valid, joint, dim])
        con[:, joint] = np.interp(all_indices, sampled_indices, sampled_con[:, joint])

    print(f"Done pose estimation {len(sampled_indices)} frames processed, {total_frames} interpolated")
    return xy, con

            
