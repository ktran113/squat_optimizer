"""
Scores the squat pipeline against hand-labeled ground truth.

Reads eval/labels.csv, runs the same pipeline as /analyze-video over each video,
and reports rep-counting and depth-classification accuracy.

Pose and detection outputs are cached per video, so re-running after changing a
threshold in squat_metrics.py rescores in seconds instead of re-running YOLO.

usage:
    python eval/run_eval.py
    python eval/run_eval.py --videos-dir ~/squat_videos --no-cache
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "backend" / "data" / "src"
BARBELL_WEIGHTS = REPO_ROOT / "backend" / "data" / "models" / "barbell" / "weights" / "best.pt"
EVAL_DIR = REPO_ROOT / "eval"

sys.path.insert(0, str(SRC_DIR))

from squat_metrics import analyze_squat  # noqa: E402
from barbell_detection import run_detection  # noqa: E402
from detect_pose import run_pose  # noqa: E402
from smooth import smooth  # noqa: E402

sys.path.insert(0, str(EVAL_DIR))
import view_check  # noqa: E402

REQUIRED_KEYPOINTS = [11, 12, 13, 14, 15, 16]
CONF_THRESHOLD = 0.5
DEPTH_CLASSES = ["below", "parallel", "partial"]
CONDITION_COLUMNS = ["distance", "angle", "lighting"]  # optional tags for per-condition slices


def video_fps(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return int(round(fps)) if fps and fps > 0 else 30


def load_labels(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("video") or row["video"].startswith("#"):
                continue
            grades = [g.strip() for g in (row.get("depth_grades") or "").split("|") if g.strip()]
            bad = [g for g in grades if g not in DEPTH_CLASSES]
            if bad:
                raise SystemExit(f"{row['video']}: unknown depth grade(s) {bad}, expected {DEPTH_CLASSES}")
            entry = {
                "video": row["video"],
                "true_reps": int(row["true_reps"]),
                "depth_grades": grades,
            }
            for col in CONDITION_COLUMNS:
                entry[col] = (row.get(col) or "").strip() or "unspecified"
            rows.append(entry)
    return rows


def pipeline_arrays(video_path, cache_dir, use_cache):
    """runs pose + detection + smoothing, caching the result per video"""
    cache_file = cache_dir / f"{video_path.stem}.npz" if cache_dir else None

    if use_cache and cache_file and cache_file.exists():
        z = np.load(cache_file)
        return z["xy"], z["conf"], z["barbell_xy"]

    raw_barbell_xy, barbell_conf = run_detection(str(video_path), str(BARBELL_WEIGHTS))
    raw_xy, conf = run_pose(str(video_path))

    xy = raw_xy.copy()
    for joint in REQUIRED_KEYPOINTS:
        xy[:, joint, :] = smooth(raw_xy[:, joint, :], conf[:, joint] > CONF_THRESHOLD)
    barbell_xy = smooth(raw_barbell_xy, barbell_conf > CONF_THRESHOLD)

    if cache_file:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, xy=xy, conf=conf, barbell_xy=barbell_xy)

    return xy, conf, barbell_xy


def rep_count_report(results):
    errors = np.array([r["pred_reps"] - r["true_reps"] for r in results])
    exact = int(np.sum(errors == 0))
    return {
        "n": len(results),
        "exact": exact,
        "accuracy": exact / len(results) if results else 0.0,
        "mae": float(np.mean(np.abs(errors))) if len(errors) else 0.0,
        "over": int(np.sum(errors > 0)),
        "under": int(np.sum(errors < 0)),
        "worst": max(results, key=lambda r: abs(r["pred_reps"] - r["true_reps"]), default=None),
    }


def depth_report(results):
    """
    Only scores videos where the rep count was exactly right, since a wrong
    count leaves no unambiguous way to pair predicted reps with labeled ones.
    """
    matrix = Counter()
    scored_videos = 0
    for r in results:
        if not r["depth_grades"] or r["pred_reps"] != r["true_reps"]:
            continue
        if len(r["depth_grades"]) != len(r["pred_depths"]):
            continue
        scored_videos += 1
        for true, pred in zip(r["depth_grades"], r["pred_depths"]):
            matrix[(true, pred)] += 1

    total = sum(matrix.values())
    correct = sum(v for (t, p), v in matrix.items() if t == p)
    return {
        "videos": scored_videos,
        "reps": total,
        "accuracy": correct / total if total else None,
        "matrix": matrix,
    }


def slice_report(results, column):
    buckets = {}
    for r in results:
        buckets.setdefault(r[column], []).append(r)
    return {k: rep_count_report(v) for k, v in sorted(buckets.items())}


def format_report(results):
    lines = []
    rc = rep_count_report(results)

    lines.append("## Rep counting\n")
    lines.append(f"- Videos evaluated: **{rc['n']}**")
    lines.append(f"- Exact-match accuracy: **{rc['accuracy']:.1%}** ({rc['exact']}/{rc['n']})")
    lines.append(f"- Mean absolute error: **{rc['mae']:.2f}** reps")
    lines.append(f"- Overcounted: {rc['over']} | Undercounted: {rc['under']}\n")

    dr = depth_report(results)
    lines.append("## Depth classification\n")
    if dr["accuracy"] is None:
        lines.append("- No scorable reps yet (needs videos with depth labels *and* a correct rep count).\n")
    else:
        lines.append(f"- Accuracy: **{dr['accuracy']:.1%}** over {dr['reps']} reps "
                     f"across {dr['videos']} videos\n")
        header = "| true \\ pred | " + " | ".join(DEPTH_CLASSES) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(DEPTH_CLASSES) + 1))
        for true in DEPTH_CLASSES:
            cells = [str(dr["matrix"].get((true, pred), 0)) for pred in DEPTH_CLASSES]
            lines.append(f"| **{true}** | " + " | ".join(cells) + " |")
        lines.append("")

    for column in CONDITION_COLUMNS:
        slices = slice_report(results, column)
        if len(slices) < 2:  # nothing to compare against
            continue
        lines.append(f"## By {column}\n")
        lines.append(f"| {column} | videos | exact-match | MAE |")
        lines.append("|---|---|---|---|")
        for name, s in slices.items():
            lines.append(f"| {name} | {s['n']} | {s['accuracy']:.1%} | {s['mae']:.2f} |")
        lines.append("")

    lines.append("## Per-video\n")
    lines.append("| video | true | pred | error | " + " | ".join(CONDITION_COLUMNS) + " |")
    lines.append("|" + "---|" * (4 + len(CONDITION_COLUMNS)))
    for r in sorted(results, key=lambda r: -abs(r["pred_reps"] - r["true_reps"])):
        err = r["pred_reps"] - r["true_reps"]
        tags = " | ".join(r[c] for c in CONDITION_COLUMNS)
        lines.append(f"| {r['video']} | {r['true_reps']} | {r['pred_reps']} | {err:+d} | {tags} |")
    lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="score the pipeline against labeled videos")
    ap.add_argument("--labels", default=str(EVAL_DIR / "labels.csv"))
    ap.add_argument("--videos-dir", default=str(EVAL_DIR / "videos"))
    ap.add_argument("--out", default=str(EVAL_DIR / "RESULTS.md"))
    ap.add_argument("--no-cache", action="store_true", help="ignore cached pose/detection output")
    ap.add_argument("--fps", type=int, default=None,
                    help="override fps (default: read from each video)")
    ap.add_argument("--max-view-ratio", type=float, default=None,
                    help="exclude videos above this frontal-view score (try --calibrate first)")
    ap.add_argument("--min-pose-conf", type=float, default=None,
                    help="exclude videos below this mean pose confidence")
    ap.add_argument("--calibrate", action="store_true",
                    help="print view_ratio and pose_conf per video, then exit without scoring")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    if not labels:
        raise SystemExit(f"no rows in {args.labels}")
    if not BARBELL_WEIGHTS.exists():
        raise SystemExit(f"barbell weights not found at {BARBELL_WEIGHTS}")

    videos_dir = Path(args.videos_dir).expanduser()
    cache_dir = EVAL_DIR / "cache"

    results, excluded, calibration = [], [], []
    for i, label in enumerate(labels, 1):
        video_path = Path(label["video"]).expanduser()
        if not video_path.is_absolute():
            video_path = videos_dir / label["video"]
        if not video_path.exists():
            print(f"[{i}/{len(labels)}] SKIP {label['video']}: not found at {video_path}")
            continue

        print(f"[{i}/{len(labels)}] {label['video']}")
        xy, conf, barbell_xy = pipeline_arrays(video_path, cache_dir, not args.no_cache)

        if args.calibrate:
            calibration.append({
                "video": label["video"],
                "angle": label["angle"],
                "ratio": view_check.view_ratio(xy, conf),
                "conf": view_check.pose_conf(conf),
            })
            continue

        ok, ratio, quality, reason = view_check.check(
            xy, conf, args.max_view_ratio, args.min_pose_conf)
        if not ok:
            excluded.append({**label, "reason": reason})
            print(f"    EXCLUDED: {reason}")
            continue

        metrics = analyze_squat(xy, conf, barbell_xy, args.fps or video_fps(video_path))

        results.append({
            **label,
            "pred_reps": int(metrics["total_reps"]),
            "pred_depths": [r["depth"] for r in metrics["reps"]],
            "view_ratio": ratio,
            "pose_conf": quality,
        })
        print(f"    true={label['true_reps']} pred={metrics['total_reps']} "
              f"view_ratio={ratio:.2f} pose_conf={quality:.2f}")

    if args.calibrate:
        if not calibration:
            raise SystemExit("nothing to calibrate, check --videos-dir")
        print("\nsorted by view_ratio (side view low, frontal high)\n")
        print(f"{'video':<40} {'view_ratio':>10} {'pose_conf':>10}  labeled_angle")
        for c in sorted(calibration, key=lambda c: (np.isnan(c["ratio"]), c["ratio"])):
            ratio = "nan" if np.isnan(c["ratio"]) else f"{c['ratio']:.3f}"
            print(f"{c['video']:<40} {ratio:>10} {c['conf']:>10.2f}  {c['angle']}")
        print("\npick a cutoff where the labeled side views end, then rerun with"
              "\n  --max-view-ratio <cutoff>")
        return

    if not results:
        raise SystemExit("no videos scored, check --videos-dir")

    report = format_report(results)
    header = (f"# Evaluation results\n\nGenerated by `eval/run_eval.py` over "
              f"{len(results)} labeled videos.\n\n")
    if excluded:
        header += (f"{len(excluded)} video(s) excluded by the view check as out of scope "
                   f"for side-view analysis:\n\n")
        for e in excluded:
            header += f"- `{e['video']}` — {e['reason']}\n"
        header += "\n"
    Path(args.out).write_text(header + report)

    print("\n" + report)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
