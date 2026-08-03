"""
Converts RepCount annotations into the labels.csv format used by run_eval.py.

RepCount (https://svip-lab.github.io/dataset/RepCount_dataset.html) ships CSVs
with a video name, a rep count, and L*/R* cycle boundary columns. Only the name
and count are needed here, since run_eval scores totals rather than cycles.

Column names are auto-detected, so this should survive minor format differences
between RepCount releases. Run with --inspect first to confirm the mapping.

usage:
    python eval/import_repcount.py --inspect ~/RepCount/annotation/train.csv
    python eval/import_repcount.py ~/RepCount/annotation/*.csv -o eval/labels_repcount.csv
"""

import argparse
import csv
import sys
from pathlib import Path

NAME_KEYS = ["name", "video", "filename", "file", "vid"]
COUNT_KEYS = ["count", "reps", "num", "n_reps", "repetitions"]
CLASS_KEYS = ["type", "class", "action", "label", "category"]

OUTPUT_FIELDS = ["video", "true_reps", "depth_grades", "distance", "angle", "lighting", "notes"]


def pick_column(fieldnames, candidates):
    """exact match first, then substring, so 'video_name' still resolves"""
    lowered = {f.lower().strip(): f for f in fieldnames if f}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    for c in candidates:
        for low, original in lowered.items():
            if c in low:
                return original
    return None


def detect_columns(fieldnames):
    return {
        "name": pick_column(fieldnames, NAME_KEYS),
        "count": pick_column(fieldnames, COUNT_KEYS),
        "class": pick_column(fieldnames, CLASS_KEYS),
    }


def read_annotations(path, action, require_class):
    """returns (rows, detected_columns, n_seen)"""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        cols = detect_columns(fields)

        if not cols["name"] or not cols["count"]:
            raise SystemExit(
                f"{path}: could not find name/count columns in {fields}\n"
                f"run with --inspect to see the header, or pass --name-col / --count-col"
            )

        rows, seen = [], 0
        for row in reader:
            seen += 1
            name = (row.get(cols["name"]) or "").strip()
            raw_count = (row.get(cols["count"]) or "").strip()
            if not name or not raw_count:
                continue

            # class may live in a column, or only in the filename
            haystack = name.lower()
            if cols["class"]:
                haystack = f"{(row.get(cols['class']) or '').lower()} {haystack}"
            elif require_class:
                continue
            if action and action.lower() not in haystack:
                continue

            try:
                count = int(float(raw_count))
            except ValueError:
                continue

            rows.append({"video": name, "true_reps": count})

        return rows, cols, seen


def inspect(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        cols = detect_columns(fields)
        print(f"\n{path}")
        print(f"  columns: {fields}")
        print(f"  detected -> name={cols['name']!r} count={cols['count']!r} class={cols['class']!r}")
        for i, row in enumerate(reader):
            if i >= 3:
                break
            preview = {k: v for k, v in list(row.items())[:6]}
            print(f"  row{i}: {preview}")


def main():
    ap = argparse.ArgumentParser(description="convert RepCount annotations to labels.csv")
    ap.add_argument("csvs", nargs="+", help="RepCount annotation CSV(s), e.g. train.csv valid.csv test.csv")
    ap.add_argument("-o", "--output", default="eval/labels_repcount.csv")
    ap.add_argument("--action", default="squat", help="substring to filter action class (default: squat)")
    ap.add_argument("--all-actions", action="store_true", help="do not filter by action")
    ap.add_argument("--inspect", action="store_true", help="print detected columns and sample rows, write nothing")
    ap.add_argument("--require-class", action="store_true",
                    help="skip rows when no class column exists instead of matching on filename")
    args = ap.parse_args()

    paths = [Path(p).expanduser() for p in args.csvs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"not found: {', '.join(str(m) for m in missing)}")

    if args.inspect:
        for p in paths:
            inspect(p)
        return

    action = None if args.all_actions else args.action
    all_rows, total_seen = [], 0
    for p in paths:
        rows, cols, seen = read_annotations(p, action, args.require_class)
        total_seen += seen
        print(f"{p.name}: {len(rows)}/{seen} rows matched "
              f"(name={cols['name']!r} count={cols['count']!r} class={cols['class']!r})")
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit(
            f"no rows matched action={action!r} across {total_seen} rows.\n"
            f"try --inspect to check the class column, or --all-actions to skip filtering"
        )

    # a video can appear in multiple splits; keep one row per name
    deduped = {}
    for row in all_rows:
        deduped.setdefault(row["video"], row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in sorted(deduped.values(), key=lambda r: r["video"]):
            writer.writerow({
                "video": row["video"],
                "true_reps": row["true_reps"],
                "depth_grades": "",
                "distance": "",
                "angle": "",
                "lighting": "",
                "notes": "repcount; CONFIRM SIDE VIEW",
            })

    dropped = len(all_rows) - len(deduped)
    print(f"\nwrote {out_path} with {len(deduped)} videos"
          + (f" ({dropped} duplicate names dropped)" if dropped else ""))
    print("next: review each video and delete rows that are not a side view,")
    print("      then fill in distance/angle/lighting to get per-condition slices")


if __name__ == "__main__":
    main()
