"""
Explains why rep_count() found the number of reps it did.

Lists every candidate peak in the depth signal with its prominence and spacing,
marks which constraint rejects it, and sweeps the thresholds so you can see what
would need to change. Use when the detected count disagrees with the real one.

Shares eval/cache with run_eval.py, so a video analyzed once is instant after.

usage:
    python scripts/diagnose_reps.py squat.mp4
    python scripts/diagnose_reps.py squat.mp4 --expected 3
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks, peak_prominences

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "backend" / "data" / "src"
CACHE_DIR = REPO_ROOT / "eval" / "cache"
BARBELL_WEIGHTS = REPO_ROOT / "backend" / "data" / "models" / "barbell" / "weights" / "best.pt"

sys.path.insert(0, str(SRC_DIR))

from squat_metrics import (  # noqa: E402
    sideSelector, squat_depths, knee_angle, seconds_to_frames, rep_signal,
    MIN_SECONDS_BETWEEN_REPS, MIN_REP_PROMINENCE,
)
from barbell_detection import run_detection  # noqa: E402
from detect_pose import run_pose  # noqa: E402
from smooth import smooth  # noqa: E402

REQUIRED_KEYPOINTS = [11, 12, 13, 14, 15, 16]
CONF_THRESHOLD = 0.5


def load_arrays(video_path, fps, use_cache=True):
    cache_file = CACHE_DIR / f"{video_path.stem}.npz"
    if use_cache and cache_file.exists():
        print(f"using cached inference: {cache_file}")
        z = np.load(cache_file)
        return z["xy"], z["conf"], z["barbell_xy"]

    raw_barbell_xy, barbell_conf = run_detection(str(video_path), str(BARBELL_WEIGHTS))
    raw_xy, conf = run_pose(str(video_path))
    xy = raw_xy.copy()
    for joint in REQUIRED_KEYPOINTS:
        xy[:, joint, :] = smooth(raw_xy[:, joint, :], conf[:, joint] > CONF_THRESHOLD, fps)
    barbell_xy = smooth(raw_barbell_xy, barbell_conf > CONF_THRESHOLD, fps)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, xy=xy, conf=conf, barbell_xy=barbell_xy)
    return xy, conf, barbell_xy


def candidate_table(depth, angles, fps):
    """
    Every candidate rep bottom, with the constraint that would reject it.
    Bottoms are minima of knee-hip, so the search runs on the negated signal
    exactly like rep_count does.
    """
    signal = rep_signal(depth)   # padded, so index i is frame i - 1
    min_distance = seconds_to_frames(MIN_SECONDS_BETWEEN_REPS, fps)

    peaks, _ = find_peaks(signal)
    if len(peaks) == 0:
        return [], []
    proms = peak_prominences(signal, peaks)[0]

    # emulate find_peaks' distance filter: strongest peak wins, neighbours drop
    order = np.argsort(-signal[peaks])
    kept = np.ones(len(peaks), bool)
    for i in order:
        if not kept[i]:
            continue
        for j in range(len(peaks)):
            if j != i and kept[j] and abs(peaks[j] - peaks[i]) < min_distance:
                kept[j] = False

    rows = []
    for i, (p, prom) in enumerate(zip(peaks, proms)):
        frame = int(p) - 1
        reasons = []
        if prom < MIN_REP_PROMINENCE:
            reasons.append(f"prominence {prom:.0f}<{MIN_REP_PROMINENCE}")
        if not kept[i]:
            reasons.append(f"within {min_distance}f of a deeper bottom")
        rows.append({
            "frame": frame,
            "time": frame / fps,
            "depth": float(depth[frame]),
            "prom": float(prom),
            "angle": float(angles[frame]) if frame < len(angles) else float("nan"),
            "reasons": reasons,
        })
    return rows, peaks - 1


def sweep(depth, proms_grid, seconds_grid, fps):
    signal = rep_signal(depth)
    table = []
    for prom in proms_grid:
        row = []
        for secs in seconds_grid:
            found, _ = find_peaks(signal, distance=seconds_to_frames(secs, fps),
                                  prominence=prom)
            row.append(len(found))
        table.append((prom, row))
    return table


def main():
    ap = argparse.ArgumentParser(description="explain the rep count for a video")
    ap.add_argument("video")
    ap.add_argument("--expected", type=int, default=None, help="true rep count, highlights matches")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        raise SystemExit(f"no such file: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    fps_int = int(round(fps))
    xy, conf, _ = load_arrays(video_path, fps_int, not args.no_cache)
    hip, knee, ank = sideSelector(xy, conf)
    depth = squat_depths(hip, knee)
    angles = knee_angle(hip, knee, ank)

    min_distance = seconds_to_frames(MIN_SECONDS_BETWEEN_REPS, fps_int)
    accepted, _ = find_peaks(rep_signal(depth), distance=min_distance,
                             prominence=MIN_REP_PROMINENCE)

    print(f"\nvideo: {video_path.name}  ({len(depth)} frames @ {fps:.0f}fps)")
    print(f"current settings: distance>{min_distance}f "
          f"({MIN_SECONDS_BETWEEN_REPS}s) prominence>{MIN_REP_PROMINENCE}px")
    print(f"detected {len(accepted)} reps"
          + (f", expected {args.expected}" if args.expected else ""))

    rows, _ = candidate_table(depth, angles, fps)
    print(f"\n{len(rows)} candidate rep bottoms in the depth signal:\n")
    print(f"{'frame':>7} {'time':>7} {'depth':>8} {'promin':>8} {'knee':>7}  verdict")
    for r in rows:
        verdict = "ACCEPTED" if not r["reasons"] else "rejected: " + "; ".join(r["reasons"])
        print(f"{r['frame']:>7} {r['time']:>6.1f}s {r['depth']:>8.0f} {r['prom']:>8.0f} "
              f"{r['angle']:>6.0f}d  {verdict}")

    near_misses = [r for r in rows if len(r["reasons"]) == 1]
    if near_misses:
        print("\nrejected by exactly one constraint (most likely your missing reps):")
        for r in near_misses:
            print(f"  frame {r['frame']} ({r['time']:.1f}s, knee {r['angle']:.0f}deg) "
                  f"-> {r['reasons'][0]}")

    prom_grid = [20, 40, 60, 80, 100, 120]
    secs_grid = [0.5, 1.0, 1.5, 2.0, 3.0]
    print(f"\nreps detected while sweeping thresholds"
          + (f" (target {args.expected})" if args.expected else "") + ":\n")
    print("prom\\gap  " + "".join(f"{s:>7}s" for s in secs_grid))
    for prom, counts in sweep(depth, prom_grid, secs_grid, fps_int):
        cells = []
        for c in counts:
            mark = "*" if args.expected and c == args.expected else " "
            cells.append(f"{c:>7}{mark}")
        current = " <- current prominence" if prom == MIN_REP_PROMINENCE else ""
        print(f"{prom:>8}px " + "".join(cells) + current)
    if args.expected:
        print("\n* = settings that give the expected count")


if __name__ == "__main__":
    main()
