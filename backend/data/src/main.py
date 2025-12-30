from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import numpy as np
import os 
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
from pydantic import BaseModel
from dotenv import load_dotenv
from squat_metrics import analyze_squat
from barbell_detection import run_detection
from detect_pose import run_pose
from smooth import smooth
from feedback import generate_feedback

load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT")
ROBOFLOW_VERSION = os.getenv("ROBOFLOW_VERSION")

app = FastAPI(
    title = "Squat Form Analysis",
    version = "0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# class SquatRequest(BaseModel):
#     xy: List[List[List[float]]]
#     conf: List[List[float]]
#     barbell_xy: List[List[float]]
#     fps: int= 30

@app.get("/")
def root():
    return {"message": "Squat analysis API is running."}

@app.post("/analyze-video")
async def analyze_squat_endpoint(file: UploadFile = File(...), fps: int = 30):
    try:
        with tempfile.NamedTemporaryFile(delete = False, suffix = ".mp4",) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)
    finally:
        file.file.close()

    try:
        #Getting keypooints
        print("Running barbell detection")
        raw_barbell_xy, barbell_conf = run_detection(tmp_path, ROBOFLOW_API_KEY, ROBOFLOW_PROJECT, ROBOFLOW_VERSION)

        # Takes only the keypoints we need which are the left and right
        # hips, knees, and ankles.
        REQUIRED_KEYPOINTS = [11, 12, 13 ,14 ,15 ,16] 
        print("Running pose estimation model")
        raw_xy, conf = run_pose(tmp_path)
        xy = raw_xy.copy()
        
        #Running savgol filter
        for joints in REQUIRED_KEYPOINTS:
            conf_valid = conf[:, joints] > 0.5      #smooth() expects boloean array
            xy[: , joints, : ] = smooth(raw_xy[:, joints, :], conf_valid)
        
        barbell_conf_valid = barbell_conf > 0.5     #smooth() expects boloean array
        barbell_xy = smooth(raw_barbell_xy, barbell_conf_valid)

        metrics = analyze_squat(xy, conf, barbell_xy, fps)

        #Feedback
        print("Creating feedback")
        feedback = generate_feedback(metrics)
        metrics["ai_feedback"] = feedback
        return metrics #returns as JSON
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during analysis: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
