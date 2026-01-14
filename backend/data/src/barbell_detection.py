import roboflow
import cv2
import numpy as np
import tempfile
import os
    
def run_detection(path, api_key, project_name, version, workspace=None):
    """
    Takes care of loading the model and returns the results as
    a tuple per frame of x,y coordinates and the confidence rating
    in an np.array
    """
    print("loading model")
    rf = roboflow.Roboflow(api_key= api_key)
    if workspace:
        project = rf.workspace(workspace).project(project_name)
    else:
        project = rf.workspace().project(project_name)
    model = project.version(version).model

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Video could not be opened")
    xy = []
    conf = []

    #create temp file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
    try:
        os.close(temp_fd)  #close file descriptor

        print("detecting barbell")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            x, y, c = detect_frame(model, frame, temp_path)

            xy.append([x,y])
            conf.append(c)
        cap.release()
        print("Done capturing barbell")
        return np.array(xy, dtype=np.float32), np.array(conf,dtype= np.float32)
    finally:
        #cleaning up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def detect_frame(model, frame, temp_path):
    """
    Analyzes individual frame and returns tuple consisting
    of that frames x,y coordinates and their confidence rating
    """
    cv2.imwrite(temp_path, frame)

    prediction = model.predict(temp_path, confidence=50).json()

    if prediction['predictions']:
        best = max(prediction["predictions"], key=lambda x: x["confidence"])
        x_cords = best["x"]
        y_cords = best["y"]
        con = best["confidence"]
        return (x_cords, y_cords, con)
    else:
        return (np.nan, np.nan, 0.0)