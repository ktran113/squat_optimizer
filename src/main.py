from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import numpy as np
import os 
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil

from dotenv import load_dotenv
from squat_metrics import analyze_squat
from barbell_detection import run_detection
from detect_pose import run_pose
from smooth import smooth

load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_PROJECT_NAME = os.getenv("ROBOFLOW_PROJECT_NAME")
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

class SquatRequest(BaseModel):
    xy: List[List[List[float]]]
    conf: List[List[float]]
    barbell_xy: List[List[float]]
    fps: int= 30

def run_full_pipeline(video_path: str, fps: int = 30):
    print("Running pipeline")
    print("Running pose-estimation")
    xy, conf = run_pose(video_path, ROBOFLOW_API_KEY, ROBOFLOW_PROJECT_NAME, ROBOFLOW_VERSION)
    print("Running barbell detection")
    barbell_xy = run_detection(video_path, ROBOFLOW_API_KEY,)

    xy, conf, barbell = smooth(xy, conf, barbell_xy)
    return metrics

@app.get("/")
def root():
    return {"message": "Squat analysis API is running."}

@app.post("/analyze-video")
async def analyze_squat_endpoint(file: UploadFile = File(), fps: int = 30):
    try:
        with tempfile.NamedTemporaryFile(delete = False, suffix = ".mp4",) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)
    finally:
        file.file.close()

    try:
        #Getting keypooints
        print("Running pose estimation model")
        raw_xy, conf = run_pose(tmp_path)
        print("Running barbell detection")
        raw_barbell_xy, barbell_conf = run_detection(tmp_path, ROBOFLOW_API_KEY, ROBOFLOW_PROJECT_NAME, ROBOFLOW_VERSION)

        #Running savgol filter
        xy = smooth(raw_xy, conf)
        barbell_xy = smooth(raw_barbell_xy, barbell_conf)

        metrics = analyze_squat(xy, conf, barbell_xy)
        return metrics #returns as JSON
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during analysis: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)