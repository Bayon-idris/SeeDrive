"""
tracking/run_sort.py
Step 7: run the C++ SORT tracker on saved YOLO detections.
"""

import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Folder that contains sort_tracker.pyd (built with CMake)
SORT_BUILD_DIR = ROOT / "tracking" / "sort_cpp" / "build" / "Release"
sys.path.insert(0, str(SORT_BUILD_DIR))

import sort_tracker 

DETECTIONS_DIR = ROOT / "outputs" / "detections"
RESULTS_DIR = (ROOT / "outputs" / "trackeval" / "trackers" / "mot_challenge"
               / "DETRAC-train" / "SORT" / "data")

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


def run_tracker(detections, n_frames, max_age=30, min_hits=3,
                iou_threshold=0.3, det_threshold=0.5):
    """Feed every frame to SORT, in order.
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

def save_results(results, out_path):
    """Write tracks in MOT format: frame, id, x, y, w, h, conf, -1, -1, -1"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for frame, track_id, x1, y1, x2, y2 in results:
            w, h = x2 - x1, y2 - y1                      # corners -> (x, y, w, h)
            f.write(f"{frame},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

    print(f"[OK] {len(results)} rows written -> {out_path}")


if __name__ == "__main__":
    detections = load_detections(DETECTIONS_DIR / "MVI_20011.txt")
    results = run_tracker(detections, n_frames=max(detections))

    print(f"[OK] {len(results)} tracked boxes")
    print(f"{len({r[1] for r in results})} different IDs")
    print("IDs on frame 50:", sorted(r[1] for r in results if r[0] == 50))
    print("IDs on frame 51:", sorted(r[1] for r in results if r[0] == 51))
    save_results(results, RESULTS_DIR / "MVI_20011.txt") 