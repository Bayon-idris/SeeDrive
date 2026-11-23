from collections import defaultdict, deque
from math import hypot

import cv2
import numpy as np

from configs.config import FPS

SPEED_WINDOW = 25  # frames: compare with the position 1 s earlier (25 fps)
SMOOTH_WINDOW = 10  # average of the last 10 speeds
MAX_DISTANCE = 80  # meters: farther positions are too imprecise


def load_tracks(txt_path):
    tracks = defaultdict(list)
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame, track_id = int(parts[0]), int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            tracks[track_id].append(
                (frame, x, y, x + w, y + h)
            )  # (x, y, w, h) -> corners
    for rows in tracks.values():
        rows.sort()
    return tracks


def ground_point(box, H):
    x1, y1, x2, y2 = box
    pixel = np.float32([[[(x1 + x2) / 2, y2]]])  # shape (1, 1, 2) for OpenCV
    X, Y = cv2.perspectiveTransform(pixel, H)[0, 0]
    return float(X), float(Y)


def track_speeds(rows, H):
    """One vehicle's boxes -> {frame: smoothed speed in km/h}."""
    positions = [(frame, *ground_point(box, H)) for frame, *box in rows]
    positions = [p for p in positions if p[2] <= MAX_DISTANCE]  # drop far positions

    speeds = {}
    recent = deque(maxlen=SMOOTH_WINDOW)  # the last raw speeds, for smoothing
    j = 0  # index of the "1 s earlier" position
    for frame, X, Y in positions:
        while frame - positions[j][0] > SPEED_WINDOW:
            j += 1
        f0, X0, Y0 = positions[j]
        if (
            frame - f0 < SPEED_WINDOW // 2
        ):  # less than 0.5 s of history: not reliable yet
            continue

        distance = hypot(X - X0, Y - Y0)  # meters traveled on the ground
        seconds = (frame - f0) / FPS  # real time, from frame numbers
        recent.append(distance / seconds * 3.6)  # m/s -> km/h
        speeds[frame] = sum(recent) / len(recent)
    return speeds


def speeds_per_frame(tracks, H):
    """Regroup everything by frame, for drawing.
    Returns {frame: [(track_id, box, speed_kmh or None), ...]}"""
    per_frame = defaultdict(list)
    for track_id, rows in tracks.items():
        speeds = track_speeds(rows, H)
        for frame, *box in rows:
            per_frame[frame].append((track_id, box, speeds.get(frame)))
    return per_frame
