import roboflow
import cv2
import numpy as np
    
def run_detection(path, api_key, project_name, version):
    print("loading model")
    rf = roboflow.Roboflow(api_key= api_key)
    project = rf.workspace().project(project_name)
    model = project.version(version).model

    cap = cv2.videoCapture(path)
    if not cap.isOpened():
        raise ValueError("Video could not be opened")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    barbell = []

    print("detecting barbell")
    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        barbell.append(detect_frame(model, cap))
    
    cap.release()
    print("Done")

    return np.array(barbell)



def detect_frame (model, frame):
    path = "temp.jpg"
    cv2.imwrite(path, frame)

    prediction = model.predict(path, confidence=50).json()

    if prediction['predicitons']:
        best = max(prediction["predicitons"], key=lambda x: x["confidence"])
        x_cords = best["x"]
        y_cords = best["y"]
        con = best["confidence"]
        return (x_cords, y_cords, con)
    else:
        return (np.nan, np.nan, 0.0)