from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List, Optional
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
from database import SessionLocal, init_db
from models import Session, RepMetric, User
from auth import hash_password, verify_password, create_access_token, get_current_user_id
from fastapi import Depends

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
    allow_origins=["https://squat-optimizer-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#initalize dbs on startup
@app.on_event("startup")
def startup_db():
    init_db()
    print("Database initialized!")

#response models
class RepMetricResponse(BaseModel):
    """
    Represents a single rep's metrics in our API response.
    This is what the frontend will receive for each rep.
    """
    id: int
    rep_number: int
    bottom_frame: Optional[int]
    start_frame: Optional[int]
    end_frame: Optional[int]
    knee_angle: Optional[float]
    depth_quality: Optional[str]
    bar_path_deviation: Optional[float]
    tempo: Optional[float]
    hip_heel_aligned: Optional[bool]

    class Config:
        from_attributes = True  #Allows conversion to db models to this format

class SessionResponse(BaseModel):
    """
    Represents a workout session in our API response.
    Contains summary stats and all the individual reps.
    """
    id: int
    user_id: Optional[int]
    video_path: str
    fps: int
    total_reps: int
    avg_depth: Optional[float]
    min_knee_angle: Optional[float]
    tempo: Optional[float]
    alignment: Optional[float]
    bar_dev: Optional[float]
    ai_feedback: Optional[str]
    created_at: str
    reps: List[RepMetricResponse] = []

    class Config:
        from_attributes = True

class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    created_at: str

    class Config:
        from_attributes = True

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    access_token: str
    user_id: int
    email: str
    name: str

#endpoints!!!
@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/register", response_model=AuthResponse)
def register(request: RegisterRequest):
    #register new account
    db = SessionLocal()
    try:
        # checks to see if email is already used
        existing_user = db.query(User).filter(User.email == request.email).first()
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )
        hashed_password = hash_password(request.password)

        new_user = User(
            email=request.email,
            password=hashed_password,
            name=request.name
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        # creates JWT token
        access_token = create_access_token(data={"sub": str(new_user.id)})

        return AuthResponse(
            access_token=access_token,
            user_id=new_user.id,
            email=new_user.email,
            name=new_user.name
        )
    finally:
        db.close()

@app.post("/login", response_model=AuthResponse)
def login(request: LoginRequest):
    #handles logging in needs email and password
    db = SessionLocal()
    try:
        #identifies user using email
        user = db.query(User).filter(User.email == request.email).first()

        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )

        # checks passowrd
        if not verify_password(request.password, user.password):
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )

        # create JWT token
        access_token = create_access_token(data={"sub": str(user.id)})

        return AuthResponse(
            access_token=access_token,
            user_id=user.id,
            email=user.email,
            name=user.name
        )
    finally:
        db.close()

@app.get("/users/{user_id}/sessions", response_model=List[SessionResponse])
def get_user_sessions(user_id: int, limit: int = 50, offset: int = 0, current_user_id: int = Depends(get_current_user_id)):
    #ensures only the user can check their own sessions
    if user_id != current_user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    # use user_id to get all workouts can be limited as well
    # returns sessionResponse and sorted by newest
    db = SessionLocal()
    try:
        sessions = db.query(Session)\
            .filter(Session.user_id == user_id)\
            .order_by(Session.created_at.desc())\
            .offset(offset)\
            .limit(limit)\
            .all()

        #convert from db to api format
        result = []
        for session in sessions:
            result.append(SessionResponse(
                id=session.id,
                user_id=session.user_id,
                video_path=session.video_path,
                fps=session.fps,
                total_reps=session.total_reps,
                avg_depth=session.avg_depth,
                min_knee_angle=session.min_knee_angle,
                tempo=session.tempo,
                alignment=session.alignment,
                bar_dev=session.bar_dev,
                ai_feedback=session.ai_feedback,
                created_at=str(session.created_at),
                reps=[RepMetricResponse(
                    id=rep.id,
                    rep_number=rep.rep_number,
                    bottom_frame=rep.bottom_frame,
                    start_frame=rep.start_frame,
                    end_frame=rep.end_frame,
                    knee_angle=rep.knee_angle,
                    depth_quality=rep.depth_quality,
                    bar_path_deviation=rep.bar_path_deviation,
                    tempo=rep.tempo,
                    hip_heel_aligned=rep.hip_heel_aligned
                ) for rep in session.reps]
            ))

        return result
    finally:
        db.close()

#specific session
@app.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: int, current_user_id: int = Depends(get_current_user_id)):
    #use session id to find sessionResponse obj that returns info
    db = SessionLocal()
    try:
        session = db.query(Session).filter(Session.id == session_id).first()

        #if not found
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session with ID {session_id} not found"
            )

        # verifying user
        if session.user_id != current_user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Convert to response format
        return SessionResponse(
            id=session.id,
            user_id=session.user_id,
            video_path=session.video_path,
            fps=session.fps,
            total_reps=session.total_reps,
            avg_depth=session.avg_depth,
            min_knee_angle=session.min_knee_angle,
            tempo=session.tempo,
            alignment=session.alignment,
            bar_dev=session.bar_dev,
            ai_feedback=session.ai_feedback,
            created_at=str(session.created_at),
            reps=[RepMetricResponse(
                id=rep.id,
                rep_number=rep.rep_number,
                bottom_frame=rep.bottom_frame,
                start_frame=rep.start_frame,
                end_frame=rep.end_frame,
                knee_angle=rep.knee_angle,
                depth_quality=rep.depth_quality,
                bar_path_deviation=rep.bar_path_deviation,
                tempo=rep.tempo,
                hip_heel_aligned=rep.hip_heel_aligned
            ) for rep in session.reps]
        )
    finally:
        db.close()

@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int, current_user_id: int = Depends(get_current_user_id)):
    if user_id != current_user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    #retrieve user profile
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            raise HTTPException(
                status_code=404,
                detail=f"User with ID {user_id} not found"
            )

        return UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            created_at=str(user.created_at)
        )
    finally:
        db.close()

@app.post("/analyze-video")
async def analyze_squat_endpoint(file: UploadFile = File(...), fps: int = 30, current_user_id: int = Depends(get_current_user_id)):
    # Input validation
    if fps < 1 or fps > 240:
        raise HTTPException(status_code=400, detail="FPS must be between 1 and 240")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed formats: {', '.join(allowed_extensions)}"
        )

    try:
        with tempfile.NamedTemporaryFile(delete = False, suffix = ".mp4",) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)
    finally:
        file.file.close()

    try:
        #getting keypoints
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

        #feedback
        print("Creating feedback")
        feedback = generate_feedback(metrics)
        metrics["ai_feedback"] = feedback

        #saving to database
        db = SessionLocal()
        try:
            # Calculate summary statistics
            avg_depth = np.mean([rep.get('bottom_angle', 0) for rep in metrics['reps']])
            min_knee_angle = min([rep.get('bottom_angle', 180) for rep in metrics['reps']])
            avg_tempo = np.mean(metrics['tempo_per_rep']) if len(metrics['tempo_per_rep']) > 0 else None
            avg_alignment = np.mean(metrics['hip_heel_alignment'])
            avg_bar_dev = np.nanmean(metrics['bar_path_dev']) if metrics['bar_path_dev'] else None

            #creates session record
            session = Session(
                user_id=current_user_id,
                video_path=tmp_path,
                fps=fps,
                total_reps=metrics['total_reps'],
                avg_depth=float(avg_depth) if not np.isnan(avg_depth) else None,
                min_knee_angle=float(min_knee_angle) if not np.isnan(min_knee_angle) else None,
                tempo=float(avg_tempo) if avg_tempo and not np.isnan(avg_tempo) else None,
                alignment=float(avg_alignment) if not np.isnan(avg_alignment) else None,
                bar_dev=float(avg_bar_dev) if avg_bar_dev and not np.isnan(avg_bar_dev) else None,
                ai_feedback=feedback
            )
            db.add(session)
            db.flush()  #get session.id

            #make metric records
            for i, rep in enumerate(metrics['reps']):
                tempo = metrics['tempo_per_rep'][i] if i < len(metrics['tempo_per_rep']) else None
                bar_dev = metrics['bar_path_dev'][i] if i < len(metrics['bar_path_dev']) else None
                hip_aligned = bool(np.mean(metrics['hip_heel_alignment'][rep['start']:rep['end']]) > 0.5)

                rep_metric = RepMetric(
                    session_id=session.id,
                    rep_number=rep['rep_count'],
                    bottom_frame=int(rep['bottom_frame']),
                    start_frame=int(rep['start']),
                    end_frame=int(rep['end']),
                    knee_angle=float(rep['bottom_angle']),
                    depth_value=None,  #can be calculated if needed
                    depth_quality=rep['depth'],
                    bar_path_deviation=float(bar_dev) if bar_dev and not np.isnan(bar_dev) else None,
                    tempo=float(tempo) if tempo and not np.isnan(tempo) else None,
                    hip_heel_aligned=hip_aligned
                )
                db.add(rep_metric)

            db.commit()
            print(f"Session {session.id} saved to database")
        except Exception as db_error:
            db.rollback()
            print(f"Database error: {db_error}")
            #continues anyways even if there is a save fail
        finally:
            db.close()

        return metrics #returns as JSON
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during analysis: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)