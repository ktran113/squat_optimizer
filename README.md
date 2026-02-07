Squat Optimizer

A computer vision system that analyzes weightlifting videos to automatically count reps and evaluate squat form.

Live Demo: [squat-optimizer-frontend.vercel.app](https://squat-optimizer-frontend.vercel.app/)

Features

1.) Automatic Rep Counting — Detects squat repetitions using signal processing with prominence-based peak detection to      filter out noise from shakiness
2.) Depth Analysis — Evaluates each rep as "below parallel," "parallel," or "partial" based on knee angle
3.) Bar Path Tracking — Measures horizontal deviation of the barbell during each rep using Roboflow object detection
4.) Hip-Heel Alignment — Checks vertical alignment between hips and ankles throughout the movement
5.) Rep Tempo — Calculates timing between reps
6.) AI Feedback — Generates coaching feedback based on the analysis

 Tech Stack
Backend:
    FastAPI (REST API)
    YOLOv8 (pose estimation)
    Roboflow (barbell detection)
    SciPy (signal processing, Savitzky-Golay smoothing)
    SQLite + SQLAlchemy (database)
    JWT authentication

Frontend:
- Deployed on Vercel

1. Pose Estimation — YOLOv8 extracts 17 body keypoints (hips, knees, ankles) from each video frame
2. Barbell Detection — Roboflow model tracks barbell position across frames
3. Smoothing — Savitzky-Golay filter reduces noise in positional data while preserving signal shape
4. Rep Detection — `scipy.signal.find_peaks` identifies rep bottoms using prominence filtering to distinguish real reps from noise
5. Metrics Calculation — Knee angles, depth quality, bar path deviation, and alignment computed per rep
6. Feedback Generation — AI generates coaching tips based on metrics

 Project Structure

```
backend/data/src/
├── main.py              # FastAPI endpoints
├── detect_pose.py       # YOLOv8 pose estimation
├── barbell_detection.py # Roboflow barbell tracking
├── squat_metrics.py     # Rep counting, angles, depth analysis
├── smooth.py            # Savitzky-Golay filtering
├── feedback.py          # AI feedback generation
├── auth.py              # JWT authentication
├── models.py            # SQLAlchemy models
└── database.py          # Database setup
```

## API Endpoints

Method  Endpoint  Description 

  POST   `/analyze-video`   Upload video, returns rep metrics and feedback  
  POST   `/register`   Create account  
  POST   `/login`   Authenticate, returns JWT  
  GET   `/users/{id}/sessions`   Get user's workout history  
  GET   `/sessions/{id}`   Get specific session details  

 Key Technical Decisions

Savitzky-Golay vs Moving Average: Used Savgol filter for smoothing because it preserves signal peaks better than moving average, which is critical for accurate rep detection. [Reference](https://medium.com/bip-xtech/stop-using-moving-average-to-smooth-your-time-series-2179af9ed59b)

Prominence-Based Peak Detection: Initial implementation overcounted reps (11 detected for 2 actual reps) due to shakiness at top/bottom of movement creating false peaks. Added `prominence` parameter to `find_peaks` to require each peak to "stand out" from surrounding signal, filtering noise while preserving real reps.
