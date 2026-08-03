"""
Renders a demo overlay (GIF or MP4) on top of a squat video showing the
pose skeleton, the tracked bar path, and per-rep metrics as they are computed.

Runs the same pipeline as /analyze-video so what you see is what the API scores.

usage:
    python scripts/render_overlay.py squat.mp4 -o demo.gif
    python scripts/render_overlay.py squat.mp4 -o demo.mp4 --start 2 --duration 6
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "backend" / "data" / "src"
BARBELL_WEIGHTS = REPO_ROOT / "backend" / "data" / "models" / "barbell" / "weights" / "best.pt"

sys.path.insert(0, str(SRC_DIR))

from squat_metrics import analyze_squat, COCO  # noqa: E402
from barbell_detection import run_detection  # noqa: E402
from detect_pose import run_pose  # noqa: E402
from smooth import smooth  # noqa: E402

REQUIRED_KEYPOINTS = [11, 12, 13, 14, 15, 16]
CONF_THRESHOLD = 0.5

# torso + limbs, face keypoints skipped since they are noisy and add nothing
SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# BGR
COLOR_SKELETON = (190, 190, 190)
COLOR_ACTIVE = (170, 255, 0)
COLOR_BAR = (0, 165, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_DIM = (160, 160, 160)

TRAIL_FRAMES = 45       # how many frames of bar path stay visible
BOTTOM_FLASH_FRAMES = 6  # frames to hold the "REP N" banner after a rep bottom
SPARKLINE_HEIGHT = 60


def analyzed_side(conf):
    """
    Mirrors sideSelector: whichever side has higher mean keypoint confidence
    is the one the metrics were computed from, so highlight that leg.
    """
    left = [COCO["L_hip"], COCO["L_knee"], COCO["L_ank"]]
    right = [COCO["R_hip"], COCO["R_knee"], COCO["R_ank"]]
    return left if np.mean(conf[:, left]) > np.mean(conf[:, right]) else right


def run_pipeline(video_path):
    """runs the same steps as the analyze-video endpoint"""
    print("Running barbell detection")
    raw_barbell_xy, barbell_conf = run_detection(video_path, str(BARBELL_WEIGHTS))

    print("Running pose estimation")
    raw_xy, conf = run_pose(video_path)
    xy = raw_xy.copy()

    for joint in REQUIRED_KEYPOINTS:
        conf_valid = conf[:, joint] > CONF_THRESHOLD
        xy[:, joint, :] = smooth(raw_xy[:, joint, :], conf_valid)

    barbell_valid = barbell_conf > CONF_THRESHOLD
    barbell_xy = smooth(raw_barbell_xy, barbell_valid)

    return xy, conf, barbell_xy


def point(xy, frame_idx, joint):
    """returns an int pixel tuple, or None if the keypoint is unusable"""
    p = xy[frame_idx, joint]
    if not np.all(np.isfinite(p)) or (p[0] == 0 and p[1] == 0):
        return None
    return int(round(p[0])), int(round(p[1]))


def draw_skeleton(frame, xy, frame_idx, active_joints, scale):
    thin = max(1, int(round(2 * scale)))
    thick = max(2, int(round(3 * scale)))

    for a, b in SKELETON_EDGES:
        pa, pb = point(xy, frame_idx, a), point(xy, frame_idx, b)
        if pa is None or pb is None:
            continue
        on_active = a in active_joints and b in active_joints
        color = COLOR_ACTIVE if on_active else COLOR_SKELETON
        cv2.line(frame, pa, pb, color, thick if on_active else thin, cv2.LINE_AA)

    for joint in range(17):
        if joint < 5:
            continue
        p = point(xy, frame_idx, joint)
        if p is None:
            continue
        active = joint in active_joints
        radius = max(2, int(round((4 if active else 3) * scale)))
        cv2.circle(frame, p, radius, COLOR_ACTIVE if active else COLOR_SKELETON, -1, cv2.LINE_AA)


def draw_bar_path(frame, barbell_xy, frame_idx, scale):
    """fading trail behind the bar plus a marker at its current position"""
    start = max(0, frame_idx - TRAIL_FRAMES)
    trail = barbell_xy[start:frame_idx + 1]

    prev = None
    for i, p in enumerate(trail):
        if not np.all(np.isfinite(p)):
            prev = None
            continue
        cur = (int(round(p[0])), int(round(p[1])))
        if prev is not None:
            fade = (i + 1) / len(trail)
            color = tuple(int(c * fade) for c in COLOR_BAR)
            cv2.line(frame, prev, cur, color, max(1, int(round(2 * scale))), cv2.LINE_AA)
        prev = cur

    if prev is not None:
        cv2.circle(frame, prev, max(3, int(round(6 * scale))), COLOR_BAR, -1, cv2.LINE_AA)
        cv2.circle(frame, prev, max(5, int(round(9 * scale))), COLOR_BAR, max(1, int(round(scale))), cv2.LINE_AA)


def draw_hud(frame, metrics, frame_idx, scale):
    """rep count, live knee angle, and the grade of the most recent rep"""
    reps = metrics["reps"]
    knee_series = metrics["knee_angle"]

    completed = [r for r in reps if r["bottom_frame"] <= frame_idx]
    knee = knee_series[frame_idx] if frame_idx < len(knee_series) else np.nan

    lines = [
        (f"REPS  {len(completed)}", COLOR_TEXT),
        (f"KNEE  {knee:.0f} deg" if np.isfinite(knee) else "KNEE  --", COLOR_DIM),
    ]
    if completed:
        lines.append((f"LAST  {completed[-1]['depth'].upper()}", COLOR_ACTIVE))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 * scale
    thickness = max(1, int(round(1.5 * scale)))
    pad = int(round(12 * scale))
    line_h = int(round(26 * scale))

    box_w = int(round(190 * scale))
    box_h = pad * 2 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, (text, color) in enumerate(lines):
        y = pad + pad + line_h * i + int(round(14 * scale))
        cv2.putText(frame, text, (pad * 2, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_rep_banner(frame, metrics, frame_idx, scale):
    """flashes when a rep bottom is reached"""
    for rep in metrics["reps"]:
        bottom = rep["bottom_frame"]
        if bottom <= frame_idx < bottom + BOTTOM_FLASH_FRAMES:
            h, w = frame.shape[:2]
            text = f"REP {rep['rep_count']}  -  {rep['depth'].upper()}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9 * scale
            thickness = max(2, int(round(2 * scale)))
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x = (w - tw) // 2
            y = h - int(round((SPARKLINE_HEIGHT + 30) * scale))
            cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
            cv2.putText(frame, text, (x, y), font, font_scale, COLOR_ACTIVE, thickness, cv2.LINE_AA)
            return


def draw_sparkline(frame, metrics, frame_idx, scale):
    """
    Depth signal along the bottom with detected rep bottoms marked. This is the
    signal the peak detector actually runs on, so it shows the counting logic.
    """
    depth = np.asarray(metrics["depth_over_time"], dtype=np.float32)
    finite = depth[np.isfinite(depth)]
    if len(finite) < 2:
        return

    h, w = frame.shape[:2]
    strip_h = int(round(SPARKLINE_HEIGHT * scale))
    top = h - strip_h
    pad_x = int(round(20 * scale))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, top), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lo, hi = float(finite.min()), float(finite.max())
    span = max(hi - lo, 1e-6)
    plot_w = w - pad_x * 2
    base = top + int(round(10 * scale))
    plot_h = strip_h - int(round(20 * scale))

    def to_px(i, value):
        x = pad_x + int(round(i / max(len(depth) - 1, 1) * plot_w))
        # signal grows as the lifter descends, so invert for a natural dip
        y = base + int(round((1 - (value - lo) / span) * plot_h))
        return x, y

    pts = [to_px(i, v) for i, v in enumerate(depth) if np.isfinite(v)]
    if len(pts) >= 2:
        cv2.polylines(frame, [np.array(pts, np.int32)], False, (110, 110, 110),
                      max(1, int(round(scale))), cv2.LINE_AA)
        upto = [p for i, p in enumerate(pts) if i <= frame_idx]
        if len(upto) >= 2:
            cv2.polylines(frame, [np.array(upto, np.int32)], False, COLOR_BAR,
                          max(1, int(round(1.5 * scale))), cv2.LINE_AA)

    for rep in metrics["reps"]:
        b = int(rep["bottom_frame"])
        if b < len(depth) and np.isfinite(depth[b]):
            seen = b <= frame_idx
            cv2.circle(frame, to_px(b, depth[b]), max(2, int(round(4 * scale))),
                       COLOR_ACTIVE if seen else (90, 90, 90), -1, cv2.LINE_AA)

    if frame_idx < len(depth) and np.isfinite(depth[frame_idx]):
        x, _ = to_px(frame_idx, depth[frame_idx])
        cv2.line(frame, (x, top + int(round(6 * scale))), (x, h - int(round(6 * scale))),
                 (255, 255, 255), max(1, int(round(scale))), cv2.LINE_AA)


def render(video_path, out_path, metrics, xy, conf, barbell_xy,
           width, target_fps, start_sec, duration_sec):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"could not open {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / target_fps)))
    out_fps = src_fps / step

    first = int(round(start_sec * src_fps))
    last = int(round((start_sec + duration_sec) * src_fps)) if duration_sec else None

    active_joints = set(analyzed_side(conf))
    n_frames = len(xy)

    writer = None
    gif_frames = []
    is_gif = out_path.suffix.lower() == ".gif"

    idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok or idx >= n_frames:
            break
        if idx < first or (last is not None and idx >= last) or (idx - first) % step != 0:
            idx += 1
            continue

        draw_skeleton(frame, xy, idx, active_joints, 1.0)
        draw_bar_path(frame, barbell_xy, idx, 1.0)

        h, w = frame.shape[:2]
        if width and w != width:
            frame = cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)

        scale = frame.shape[1] / 640.0
        draw_sparkline(frame, metrics, idx, scale)
        draw_hud(frame, metrics, idx, scale)
        draw_rep_banner(frame, metrics, idx, scale)

        if is_gif:
            from PIL import Image
            gif_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, out_fps,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)

        kept += 1
        idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    if is_gif:
        if not gif_frames:
            raise SystemExit("no frames rendered, check --start / --duration")
        gif_frames[0].save(
            out_path, save_all=True, append_images=gif_frames[1:],
            duration=int(round(1000 / out_fps)), loop=0, optimize=True,
        )

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nwrote {out_path} ({kept} frames, {out_fps:.1f}fps, {size_mb:.1f}MB)")
    if is_gif and size_mb > 10:
        print("note: over 10MB is large for a README, try --duration 4 --width 480")


def main():
    ap = argparse.ArgumentParser(description="render a pose + bar path overlay demo")
    ap.add_argument("video", help="input squat video")
    ap.add_argument("-o", "--output", default="demo.gif", help="output .gif or .mp4")
    ap.add_argument("--width", type=int, default=640, help="output width in px")
    ap.add_argument("--fps", type=float, default=12, help="target output fps")
    ap.add_argument("--start", type=float, default=0, help="start time in seconds")
    ap.add_argument("--duration", type=float, default=None, help="clip length in seconds")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        raise SystemExit(f"no such file: {args.video}")
    if not BARBELL_WEIGHTS.exists():
        raise SystemExit(f"barbell weights not found at {BARBELL_WEIGHTS}")

    xy, conf, barbell_xy = run_pipeline(args.video)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    metrics = analyze_squat(xy, conf, barbell_xy, int(round(fps)))
    print(f"\ndetected {metrics['total_reps']} reps: "
          f"{', '.join(r['depth'] for r in metrics['reps']) or 'none'}")

    render(args.video, Path(args.output), metrics, xy, conf, barbell_xy,
           args.width, args.fps, args.start, args.duration)


if __name__ == "__main__":
    main()
