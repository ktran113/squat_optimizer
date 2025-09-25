def run_pose(video_path, weights='yolov8s-pose.pt'):
    from ultralytics import YOLO
    import numpy as np

    model = YOLO(weights)
    results = model(video_path)

    xy = []     #Stores x-y coordinates
    con = []    #Stores confidence of those coordinates

    for frame in results:
        #Dealing with frames where keypoints cant be detected. 
        if frame.keypoints is None or len(frame.keypoints) == 0:
            xy.append(np.full((17,2), np.nan, dtype=np.float32))
            con.append(np.zeros((17, 0), dtype= np.float32))
            continue

        #Case where frames are detectable
        person_idx = int(frame.boxes.conf.argmax().item())
        xy.append(frame.keypoints.xy[person_idx].cpu().numpy().astype(np.float32))
        con.append(frame.keypoints.conf[person_idx].cpu().numpy().astype(np.float32))

    return np.stack(xy), np.stack(con)

            
