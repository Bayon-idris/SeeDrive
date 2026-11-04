"""
tracking/evaluate.py
Step 8: score tracker results against the UA-DETRAC ground truth with TrackEval.

Run from the project root:
    python -m tracking.evaluate
"""

import shutil

import numpy as np

# TrackEval (2022) still uses np.float / np.int, removed in recent numpy.
# Re-create them BEFORE importing trackeval. (Do not touch np.bool.)
np.float = float
np.int = int

import trackeval  # noqa: E402
from types import SimpleNamespace

from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


from configs.config import BENCHMARK, GT_FOLDER, ROOT, SPLIT, TRACKERS_FOLDER  # noqa: E402

                  # folders are named DETRAC-train


BYTETRACK_ARGS = SimpleNamespace(
    track_high_thresh=0.5,   # 1st pass: same threshold as SORT -> fair comparison
    track_low_thresh=0.1,    # 2nd pass: low-confidence boxes get a second chance
    new_track_thresh=0.6,    # a new vehicle needs a confident detection
    track_buffer=30,         # frames a lost vehicle is kept (same as SORT max_age)
    match_thresh=0.8,
    fuse_score=True,
)

def make_gt_copy(sequences):
    """Sanity check tracker: the ground truth itself, saved as a fake tracker.
    Scoring it must give 100 everywhere."""
    out_dir = TRACKERS_FOLDER / f"{BENCHMARK}-{SPLIT}" / "GT_COPY" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    for seq in sequences:
        gt_file = GT_FOLDER / f"{BENCHMARK}-{SPLIT}" / seq / "gt" / "gt.txt"
        shutil.copy(gt_file, out_dir / f"{seq}.txt")
    print(f"[OK] GT_COPY tracker created for {len(sequences)} sequence(s)")


def evaluate(trackers, sequences):
    """Score each tracker on the given sequences. Returns TrackEval's raw results."""
    evaluator = trackeval.Evaluator({
        "USE_PARALLEL": False,
        "PRINT_CONFIG": False,
        "PRINT_RESULTS": True,       # prints the HOTA / CLEAR / Identity tables
        "OUTPUT_SUMMARY": False,
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
    })
    dataset = trackeval.datasets.MotChallenge2DBox({
        "GT_FOLDER": str(GT_FOLDER),
        "TRACKERS_FOLDER": str(TRACKERS_FOLDER),
        "BENCHMARK": BENCHMARK,
        "SPLIT_TO_EVAL": SPLIT,
        "TRACKERS_TO_EVAL": trackers,
        "SEQ_INFO": {seq: None for seq in sequences},   # None = read length from seqinfo.ini
        "DO_PREPROC": False,         # ignored regions are already removed by run_sort.py
        "PRINT_CONFIG": False,
    })
    quiet = {"PRINT_CONFIG": False}
    metrics = [trackeval.metrics.HOTA(quiet),        # HOTA: detection + identity, balanced
               trackeval.metrics.CLEAR(quiet),       # MOTA, ID switches (IDSW)
               trackeval.metrics.Identity(quiet)]    # IDF1

    results, _ = evaluator.evaluate([dataset], metrics)
    return results["MotChallenge2DBox"]

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


# if __name__ == "__main__":
#     sequences = ["MVI_20011"]
#     evaluate(["SORT"], sequences)

if __name__ == "__main__":
    sequences = ["MVI_20011"]
    evaluate(["SORT", "BYTETRACK"], sequences)