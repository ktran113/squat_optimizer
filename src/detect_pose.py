def run_pose(video_path, weights='yolov8s-pose.pt'):
    from ultralytics import YOLO
    import numpy as np

    model = YOLO(weights) 
    xy, cf = [], []
    for res in model.track(source=video_path, stream=True, verbose=False, persist=True):
        if res.keypoints is None or len(res.keypoints) == 0:
            #deals with frames where a keypoint cant be detected
            xy.append(np.full((17,2), np.nan, np.float32))
            cf.append(np.zeros(17, np.float32))
            continue
        i = int(res.boxes.conf.argmax())  # i is "person" w highest confidence rating
        xy.append(res.keypoints.xy[i].cpu().numpy().astype('float32'))
        cf.append(res.keypoints.conf[i].cpu().numpy().astype('float32'))
    return np.stack(xy), np.stack(cf)