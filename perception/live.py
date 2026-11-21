"""
perception/live.py
SeeDrive live pipeline for ONE camera:
RTSP stream -> fine-tuned YOLO -> C++ SORT -> traffic logic (speeds, counts, rules) -> alerts.

    Thread 1 (reader)     : reads the RTSP stream, keeps only the latest frame + its number
    Thread 2 (perception) : YOLO detection + SORT tracking on the latest frame
    Main thread           : traffic logic, drawing, display

Run from the project root (MediaMTX must be running, cam_01 must be enabled in cameras.yaml):
    python -m perception.live --camera cam_01
    python -m perception.live --camera cam_01 --simulate     # demo: wrong-way alerts
"""

import argparse
import os
import threading
import time

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")   # before importing cv2

import cv2
import numpy as np
from ultralytics import YOLO

from configs.config import CONFIG_PATH, ROOT
import sort_tracker  # noqa: E402  (configs.config adds the C++ build folder to sys.path)
from ingest.rtsp_reader import load_cameras, start_cameras, stop_cameras
from logic.engine import TrafficLogic, draw_label

MODEL_PATH = ROOT / "runs" / "yolo11n_detrac" / "weights" / "best.pt"


class Latest:
    """A box with one place, shared by two threads (same idea as SharedSlot)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = None
        self.version = 0                  # increases at each put(): tells the reader it's new

    def put(self, data):
        with self.lock:
            self.data = data
            self.version += 1

    def get(self):
        with self.lock:
            return self.data, self.version


# ------------------------------------------------------------------ thread 1: reader
def reader_loop(url, frames, stop):
    for _ in range(10):                   # the publisher may need a few seconds to start
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            break
        cap.release()
        time.sleep(1)
    else:
        print(f"[ERROR] Cannot open {url}")
        stop.set()
        return

    frame_no = 0
    while not stop.is_set():
        ok, image = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        frame_no += 1                     # counts EVERY frame read: real time = frame_no / FPS
        frames.put((frame_no, image))
    cap.release()


# ------------------------------------------------------------------ thread 2: perception
def perception_loop(model, frames, results, stop):
    tracker = sort_tracker.Sort(max_age=30, min_hits=3, iou_threshold=0.3, det_threshold=0.5)
    last_version = 0
    while not stop.is_set():
        data, version = frames.get()
        if data is None or version == last_version:
            time.sleep(0.001)             # no new frame yet
            continue
        last_version = version
        frame_no, image = data

        result = model.predict(image, imgsz=960, conf=0.1, verbose=False)[0]      # detection
        dets = np.hstack([result.boxes.xyxy.cpu().numpy(),                       # (N, 5):
                          result.boxes.conf.cpu().numpy()[:, None]])            # x1 y1 x2 y2 score
        tracks = [(int(t[4]), tuple(t[:4])) for t in tracker.update(dets)]     # C++ SORT: + IDs
        results.put((frame_no, image, tracks))


# ------------------------------------------------------------------ main thread
def run(camera_id, simulate=False):
    """Run the full live pipeline on one camera (until 'q' or Ctrl+C)."""
    camera = next((c for c in load_cameras(CONFIG_PATH) if c["id"] == camera_id), None)
    if camera is None:
        print(f"[ERROR] {camera_id} not found or not enabled in {CONFIG_PATH}")
        return

    model = YOLO(str(MODEL_PATH))
    logic = TrafficLogic(camera_id, simulate=simulate)
    frames, results, stop = Latest(), Latest(), threading.Event()

    processes = start_cameras([camera])                       # FFmpeg publishes the simulated camera
    threads = [
        threading.Thread(target=reader_loop, args=(camera["rtsp_url"], frames, stop), daemon=True),
        threading.Thread(target=perception_loop, args=(model, frames, results, stop), daemon=True),
    ]
    for t in threads:
        t.start()

    last_version = 0
    fps, fps_frames, fps_t0 = 0.0, 0, time.perf_counter()
    try:
        while not stop.is_set():
            data, version = results.get()
            if data is None or version == last_version:
                time.sleep(0.001)
                continue
            last_version = version
            frame_no, image, tracks = data

            logic.update(frame_no, tracks)                    # speeds, counts, rules, alerts
            logic.draw(image, frame_no)

            fps_frames += 1                                   # processed frames per second
            now = time.perf_counter()
            if now - fps_t0 >= 1.0:
                fps, fps_frames, fps_t0 = fps_frames / (now - fps_t0), 0, now
            draw_label(image, f"{camera_id} | {fps:.1f} FPS", 10, 55, scale=0.6)

            cv2.imshow("SeeDrive - live", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=2)
        stop_cameras(processes)
        cv2.destroyAllWindows()
        print("Counts:", logic.counter.counts)