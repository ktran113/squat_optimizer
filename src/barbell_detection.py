import roboflow
import cv2
import numpy as np
    
def run_detection(path, api_key, project_name, version):
    """
    Takes care of loading the model and returns the results as
    a tuple per frame of x,y coordinates and the confidence rating
    in an np.array
    """
    print("loading model")
    rf = roboflow.Roboflow(api_key= api_key)
    project = rf.workspace().project(project_name)
    model = project.version(version).model

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Video could not be opened")
    xy = []
    conf = []

    print("detecting barbell")
    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        x, y, c = detect_frame(model, frame)

        xy.append([x,y])
        conf.append(c)
    cap.release()
    print("Done using roboflow workflow to capture barbell")

    return np.array(xy, dtype=np.float32), np.array(conf,dtype= np.float32)

def detect_frame (model, frame):
    """
    Analyzes individual frame and returns tuple consisting
    of that frames x,y cordinates and their confidence rating
    """
    path = "temp.jpg"
    cv2.imwrite(path, frame)

    prediction = model.predict(path, confidence=50).json()

    if prediction['predictions']:
        best = max(prediction["predictions"], key=lambda x: x["confidence"])
        x_cords = best["x"]
        y_cords = best["y"]
        con = best["confidence"]
        return (x_cords, y_cords, con)
    else:
        return (np.nan, np.nan, 0.0)