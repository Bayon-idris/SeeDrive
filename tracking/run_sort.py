"""
tracking/run_sort.py
Run two trackers on the saved YOLO detections:
    - SORT      : our C++ tracker (sort_tracker module, built with pybind11)
    - ByteTrack : Ultralytics' implementation, used as a reference
Results are written in MOT format for TrackEval (tracking/evaluate.py).

Run from the project root:
    python -m tracking.run_sort
"""

import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from configs.config import DETECTIONS_DIR, ROOT, TRACKERS_DIR
from tracking.convert_gt import IGNORED_DIR

# Folder that contains sort_tracker.pyd (built with CMake)
SORT_BUILD_DIR = ROOT / "tracking" / "sort_cpp" / "build" / "Release"
sys.path.insert(0, str(SORT_BUILD_DIR))

import sort_tracker  # noqa: E402  (import after sys.path change)

SEQUENCES = ["MVI_20011"]

BYTETRACK_ARGS = SimpleNamespace(
    track_high_thresh=0.5,   # 1st pass: same threshold as SORT -> fair comparison
    track_low_thresh=0.1,    # 2nd pass: low-confidence boxes get a second chance
    new_track_thresh=0.6,    # a new vehicle needs a confident detection
    track_buffer=30,         # frames a lost vehicle is kept (same as SORT max_age)
    match_thresh=0.8,
    fuse_score=True,
)


# ------------------------------------------------------------------ input
def load_detections(txt_path):
    """Read a MOT detection file.
    Returns {frame: array (N, 5) [x1, y1, x2, y2, score]}"""
    rows_per_frame = defaultdict(list)

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame = int(parts[0])
            x, y, w, h, score = map(float, parts[2:7])
            rows_per_frame[frame].append([x, y, x + w, y + h, score])   # (x, y, w, h) -> corners

    return {frame: np.array(rows) for frame, rows in rows_per_frame.items()}


# ------------------------------------------------------------------ trackers
def run_tracker(detections, n_frames, max_age=30, min_hits=3,
                iou_threshold=0.3, det_threshold=0.5):
    """Feed every frame to our C++ SORT, in order.
    Returns a list of rows: (frame, id, x1, y1, x2, y2)"""
    tracker = sort_tracker.Sort(max_age=max_age, min_hits=min_hits,
                                iou_threshold=iou_threshold,
                                det_threshold=det_threshold)
    results = []
    no_boxes = np.empty((0, 5))

    for frame in range(1, n_frames + 1):
        dets = detections.get(frame, no_boxes)   # frame without detection -> empty
        tracks = tracker.update(dets)            # C++ SORT adds the IDs

        for x1, y1, x2, y2, track_id in tracks:
            results.append((frame, int(track_id), x1, y1, x2, y2))

    return results


def run_bytetrack(detections, n_frames, image_size=(540, 960)):
    """Same job as run_tracker, but with Ultralytics' ByteTrack.
    Returns a list of rows: (frame, id, x1, y1, x2, y2)"""
    tracker = BYTETracker(BYTETRACK_ARGS)
    results = []
    no_boxes = np.empty((0, 5))

    for frame in range(1, n_frames + 1):
        dets = detections.get(frame, no_boxes)
        data = np.hstack([dets, np.zeros((len(dets), 1))])   # add class 0: [x1, y1, x2, y2, score, cls]
        tracks = tracker.update(Boxes(data, image_size))

        for x1, y1, x2, y2, track_id, *_ in tracks:          # *_ = ignore score, class, index
            results.append((frame, int(track_id), x1, y1, x2, y2))

    return results


# ------------------------------------------------------------------ ignored regions
def load_ignored_regions(path):
    """Ignored regions of a sequence, as corners (x1, y1, x2, y2)."""
    regions = []
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    x, y, w, h = map(float, line.split(","))
                    regions.append((x, y, x + w, y + h))
    return regions


def inside_ignored(box, regions, min_ratio=0.5):
    """True if at least min_ratio of the box area lies inside an ignored region."""
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return False
    for rx1, ry1, rx2, ry2 in regions:
        iw = max(0.0, min(x2, rx2) - max(x1, rx1))   # same idea as the IoU intersection
        ih = max(0.0, min(y2, ry2) - max(y1, ry1))
        if iw * ih / area >= min_ratio:
            return True
    return False


def remove_ignored(results, regions):
    """Drop tracked boxes that are inside ignored regions."""
    kept = [r for r in results if not inside_ignored(r[2:6], regions)]
    print(f"[FILTER] {len(results) - len(kept)} boxes removed (ignored regions)")
    return kept


# ------------------------------------------------------------------ output
def save_results(results, out_path):
    """Write tracks in MOT format: frame, id, x, y, w, h, conf, -1, -1, -1"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for frame, track_id, x1, y1, x2, y2 in results:
            w, h = x2 - x1, y2 - y1                      # corners -> (x, y, w, h)
            f.write(f"{frame},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

    print(f"[OK] {len(results)} rows written -> {out_path}")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    for seq in SEQUENCES:
        print(f"\n=== {seq}")
        detections = load_detections(DETECTIONS_DIR / f"{seq}.txt")    # 1) read
        regions = load_ignored_regions(IGNORED_DIR / f"{seq}.txt")
        n_frames = max(detections)

        sort_results = remove_ignored(run_tracker(detections, n_frames), regions)     # 2) track + clean
        byte_results = remove_ignored(run_bytetrack(detections, n_frames), regions)

        print(f"SORT:      {len(sort_results)} boxes, {len({r[1] for r in sort_results})} IDs")
        print(f"ByteTrack: {len(byte_results)} boxes, {len({r[1] for r in byte_results})} IDs")

        save_results(sort_results, TRACKERS_DIR / "SORT" / "data" / f"{seq}.txt")    # 3) write
        save_results(byte_results, TRACKERS_DIR / "BYTETRACK" / "data" / f"{seq}.txt")