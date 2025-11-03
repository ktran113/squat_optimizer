import cv2
import numpy as np
from scipy.signal import find_peaks

# COCO keypoint connections for drawing skeleton
SKELETON_CONNECTIONS = [
    (5, 6),   # Shoulders
    (5, 7),   # Left shoulder to elbow
    (7, 9),   # Left elbow to wrist
    (6, 8),   # Right shoulder to elbow
    (8, 10),  # Right elbow to wrist
    (5, 11),  # Left shoulder to hip
    (6, 12),  # Right shoulder to hip
    (11, 12), # Hips
    (11, 13), # Left hip to knee
    (13, 15), # Left knee to ankle
    (12, 14), # Right hip to knee
    (14, 16), # Right knee to ankle
]

# COCO keypoint names
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def count_reps_from_depth(depth_over_time, min_depth_threshold=0, min_distance=30):

    # Find peaks (bottoms of squats) where depth is positive (at/below parallel)
    peaks, _ = find_peaks(depth_over_time, height=min_depth_threshold, distance=min_distance)
    return peaks


def draw_skeleton(frame, keypoints, confidences, conf_threshold=0.5, 
                 color=(0, 255, 0), thickness=2):
    """
    Draw skeleton connections on frame.
    
    Parameters:
    -----------
    frame : np.ndarray
        Video frame to draw on
    keypoints : np.ndarray, shape (17, 2)
        Keypoint positions for this frame
    confidences : np.ndarray, shape (17,)
        Keypoint confidences
    conf_threshold : float
        Minimum confidence to draw a connection
    color : tuple
        BGR color for skeleton lines
    thickness : int
        Line thickness
    """
    for connection in SKELETON_CONNECTIONS:
        pt1_idx, pt2_idx = connection
        
        # Check if both keypoints have sufficient confidence
        if confidences[pt1_idx] > conf_threshold and confidences[pt2_idx] > conf_threshold:
            pt1 = tuple(keypoints[pt1_idx].astype(int))
            pt2 = tuple(keypoints[pt2_idx].astype(int))
            
            # Draw line between keypoints
            cv2.line(frame, pt1, pt2, color, thickness)


def draw_keypoints(frame, keypoints, confidences, conf_threshold=0.5,
                  color=(0, 0, 255), radius=5, draw_labels=True):
    """
    Draw keypoints and optional labels on frame.
    
    Parameters:
    -----------
    frame : np.ndarray
        Video frame to draw on
    keypoints : np.ndarray, shape (17, 2)
        Keypoint positions
    confidences : np.ndarray, shape (17,)
        Keypoint confidences
    conf_threshold : float
        Minimum confidence to draw
    color : tuple
        BGR color for keypoints
    radius : int
        Circle radius
    draw_labels : bool
        Whether to draw keypoint names
    """
    for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
        if conf > conf_threshold:
            x, y = kp.astype(int)
            
            # Draw circle for keypoint
            cv2.circle(frame, (x, y), radius, color, -1)
            cv2.circle(frame, (x, y), radius + 2, (255, 255, 255), 1)  # White outline
            
            # Draw label if requested
            if draw_labels:
                label = KEYPOINT_NAMES[i]
                cv2.putText(frame, label, (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def draw_stats_overlay(frame, rep_count, current_rep, avg_time_per_rep, 
                       current_rep_time=None):
    """
    Draw rep counter and timing stats on frame.
    
    Parameters:
    -----------
    frame : np.ndarray
        Video frame to draw on
    rep_count : int
        Total number of reps completed
    current_rep : int
        Current rep number (0-indexed)
    avg_time_per_rep : float
        Average time per rep in seconds
    current_rep_time : float, optional
        Time elapsed in current rep
    """
    # Create semi-transparent overlay box
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Top-left stats box
    box_height = 120
    cv2.rectangle(overlay, (10, 10), (300, box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw stats text
    y_offset = 35
    
    # Rep count
    cv2.putText(frame, f"Reps: {rep_count}", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Current rep
    if current_rep >= 0:
        y_offset += 30
        cv2.putText(frame, f"Current: Rep {current_rep + 1}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # Average time per rep
    if avg_time_per_rep > 0:
        y_offset += 30
        cv2.putText(frame, f"Avg Time: {avg_time_per_rep:.1f}s", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # Current rep time (live timer)
    if current_rep_time is not None:
        y_offset += 30
        cv2.putText(frame, f"This Rep: {current_rep_time:.1f}s", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)


def create_overlay_video(video_path, xy, con, output_path, 
                         squat_depth_data=None, fps=30,
                         draw_labels=False, conf_threshold=0.5):
    """
    Create video with pose keypoints overlaid and rep counter.
    
    Parameters:
    -----------
    video_path : str
        Input video path
    xy : np.ndarray, shape (num_frames, 17, 2)
        Keypoint positions
    con : np.ndarray, shape (num_frames, 17)
        Keypoint confidences
    output_path : str
        Output video path
    squat_depth_data : dict, optional
        Output from squat_depth() function containing 'depth_over_time'
    fps : int
        Frames per second
    draw_labels : bool
        Whether to draw keypoint labels
    conf_threshold : float
        Minimum confidence to draw keypoints
    """
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Count reps if depth data provided
    rep_frames = np.array([])
    rep_count = 0
    rep_times = []
    
    if squat_depth_data is not None:
        depth_over_time = squat_depth_data['depth_over_time']
        rep_frames = count_reps_from_depth(depth_over_time)
        rep_count = len(rep_frames)
        
        # Calculate time for each rep
        if len(rep_frames) > 0:
            for i in range(len(rep_frames) - 1):
                rep_duration = (rep_frames[i + 1] - rep_frames[i]) / fps
                rep_times.append(rep_duration)
    
    avg_time_per_rep = np.mean(rep_times) if len(rep_times) > 0 else 0
    
    print(f"Processing video: {total_frames} frames")
    print(f"Found {rep_count} reps")
    if avg_time_per_rep > 0:
        print(f"Average time per rep: {avg_time_per_rep:.1f}s")
    
    frame_idx = 0
    current_rep = -1
    current_rep_start_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Determine current rep
        if len(rep_frames) > 0:
            # Find which rep we're in
            reps_completed = np.sum(rep_frames <= frame_idx)
            
            if reps_completed > 0:
                current_rep = reps_completed - 1
                if current_rep < len(rep_frames):
                    current_rep_start_frame = rep_frames[current_rep]
            else:
                current_rep = -1
        
        # Calculate current rep time
        current_rep_time = None
        if current_rep >= 0:
            current_rep_time = (frame_idx - current_rep_start_frame) / fps
        
        # Draw skeleton
        draw_skeleton(frame, xy[frame_idx], con[frame_idx], 
                     conf_threshold=conf_threshold,
                     color=(0, 255, 0), thickness=2)
        
        # Draw keypoints
        draw_keypoints(frame, xy[frame_idx], con[frame_idx],
                      conf_threshold=conf_threshold,
                      color=(0, 0, 255), radius=4,
                      draw_labels=draw_labels)
        
        # Draw stats overlay
        draw_stats_overlay(frame, rep_count, current_rep, 
                          avg_time_per_rep, current_rep_time)
        
        # Write frame
        out.write(frame)
        
        frame_idx += 1
        
        # Progress indicator
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"Progress: {progress:.1f}%", end='\r')
    
    # Clean up
    cap.release()
    out.release()
    
    print(f"\nVideo saved to: {output_path}")


def create_overlay_video_with_barbell(video_path, xy, con, bar_xy, output_path,
                                      squat_depth_data=None, fps=30,
                                      draw_labels=False, conf_threshold=0.5):
    """
    Create video with pose keypoints AND barbell tracking overlaid.
    
    Parameters:
    -----------
    bar_xy : np.ndarray, shape (num_frames, 2)
        Barbell center positions
    """
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Count reps
    rep_frames = np.array([])
    rep_count = 0
    rep_times = []
    
    if squat_depth_data is not None:
        depth_over_time = squat_depth_data['depth_over_time']
        rep_frames = count_reps_from_depth(depth_over_time)
        rep_count = len(rep_frames)
        
        if len(rep_frames) > 0:
            for i in range(len(rep_frames) - 1):
                rep_duration = (rep_frames[i + 1] - rep_frames[i]) / fps
                rep_times.append(rep_duration)
    
    avg_time_per_rep = np.mean(rep_times) if len(rep_times) > 0 else 0
    
    # Store bar path for visualization
    bar_path_history = []
    
    frame_idx = 0
    current_rep = -1
    current_rep_start_frame = 0
    
    print(f"Processing video with barbell tracking...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Determine current rep
        if len(rep_frames) > 0:
            reps_completed = np.sum(rep_frames <= frame_idx)
            if reps_completed > 0:
                current_rep = reps_completed - 1
                if current_rep < len(rep_frames):
                    current_rep_start_frame = rep_frames[current_rep]
            else:
                current_rep = -1
        
        current_rep_time = None
        if current_rep >= 0:
            current_rep_time = (frame_idx - current_rep_start_frame) / fps
        
        # Draw skeleton
        draw_skeleton(frame, xy[frame_idx], con[frame_idx], 
                     conf_threshold=conf_threshold,
                     color=(0, 255, 0), thickness=2)
        
        # Draw keypoints
        draw_keypoints(frame, xy[frame_idx], con[frame_idx],
                      conf_threshold=conf_threshold,
                      color=(0, 0, 255), radius=4,
                      draw_labels=draw_labels)
        
        # Draw barbell position and path
        if np.isfinite(bar_xy[frame_idx]).all():
            bar_x, bar_y = bar_xy[frame_idx].astype(int)
            
            # Draw barbell as larger circle
            cv2.circle(frame, (bar_x, bar_y), 8, (255, 0, 255), -1)  # Magenta
            cv2.circle(frame, (bar_x, bar_y), 10, (255, 255, 255), 2)  # White outline
            
            # Add to path history
            bar_path_history.append((bar_x, bar_y))
            
            # Draw bar path (last 60 frames = 2 seconds at 30fps)
            if len(bar_path_history) > 1:
                path_points = bar_path_history[-60:]  # Last 2 seconds
                for i in range(len(path_points) - 1):
                    cv2.line(frame, path_points[i], path_points[i + 1], 
                            (255, 100, 255), 2)
        
        # Draw stats overlay
        draw_stats_overlay(frame, rep_count, current_rep, 
                          avg_time_per_rep, current_rep_time)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"Progress: {progress:.1f}%", end='\r')
    
    cap.release()
    out.release()
    
    print(f"\nVideo saved to: {output_path}")