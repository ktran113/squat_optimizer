def run_pose(video_path, weights='yolov8s-pose.pt'):
    from ultralytics import YOLO
    import numpy as np

    model = YOLO(weights)
    results = model(video_path, stream = True)

    xy = []
    con = []
    for frame in results:
        #Dealing with frames where keypoints cant be detected. 
        if frame.keypoints is None or len(frame.keypoints) == 0:
            xy.append(np.full((17,2), np.nan, dtype=np.float32))
            con.append(np.zeros((17,), dtype= np.float32))
            continue

        #Normal case
        person_idx = int(frame.boxes.conf.argmax().item())
        xy.append(frame.keypoints.xy[person_idx].cpu().numpy().astype(np.float32))

        if frame.keypoints.conf is not None:
            con.append(frame.keypoints.conf[person_idx].cpu().numpy().astype(np.float32))
        else:
            con.append(np.zeros(17,), dtype=np.float32)

    return np.stack(xy), np.stack(con)

            
