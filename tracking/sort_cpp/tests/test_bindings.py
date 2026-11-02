
"""
Step 6 test: same scenario as test_sort.cpp, but called from Python.
The IDs must be exactly the same as in C++ (A=1, B=2, new car C=3).

Run from the sort_cpp folder:
    python tests/test_bindings.py
"""

import sys
from pathlib import Path

import numpy as np

# Find the compiled module (Windows: build/Release, Linux/macOS: build)
ROOT = Path(__file__).resolve().parent.parent
for folder in (ROOT / "build" / "Release", ROOT / "build"):
    if folder.is_dir():
        sys.path.insert(0, str(folder))

import sort_tracker  # noqa: E402

failures = 0


def check(condition, name):
    global failures
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        failures += 1


def id_near(tracks, x):
    """ID of the output box around x, or -1."""
    for x1, _, x2, _, track_id in tracks:
        if x1 - 20 < x < x2 + 20:
            return int(track_id)
    return -1


tracker = sort_tracker.Sort(max_age=5, min_hits=3, iou_threshold=0.3, det_threshold=0.5)

# Car A moves right 3 px/frame, car B is parked, one low-confidence detection (ignored)
ids_a, ids_b = set(), set()
for f in range(1, 11):
    ax = 100 + 3 * f
    dets = np.array([
        [ax, 100, ax + 50, 140, 0.9],
        [600, 300, 660, 350, 0.8],
        [300, 400, 330, 420, 0.2],
    ])
    tracks = tracker.update(dets)
    ids_a.add(id_near(tracks, ax + 25))
    ids_b.add(id_near(tracks, 630))

print(f"  ids: A={ids_a}, B={ids_b}")
check(ids_a == {1} and ids_b == {2}, "A keeps id 1, B keeps id 2")
check(tracker.num_tracks == 2, "low-confidence detection ignored (2 tracks)")

# Car A disappears for more than max_age frames
for f in range(11, 18):
    tracker.update(np.array([[600, 300, 660, 350, 0.8]]))
check(tracker.num_tracks == 1, "A deleted after max_age frames")

# New car C appears
id_c = -1
for f in range(18, 23):
    tracks = tracker.update(np.array([[600, 300, 660, 350, 0.8], [200, 50, 240, 80, 0.9]]))
    if id_near(tracks, 220) != -1:
        id_c = id_near(tracks, 220)
check(id_c == 3, "new car C gets id 3")

# Empty frame and output format
empty = tracker.update(np.empty((0, 5)))
check(empty.shape[1] == 5, "empty frame handled, output has 5 columns")

print("\nAll checks passed." if failures == 0 else f"\n{failures} check(s) failed.")
sys.exit(1 if failures else 0)