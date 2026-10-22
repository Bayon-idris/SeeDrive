import os
import time
import threading
from collections import deque

from configs.config import STAGES, TARGET_CLASSES
from perception.latency import LatencyStats
from perception.shared_slot import SharedSlot
from perception.helpers import load_model

# RTSP over TCP (no lost packets). Must be set BEFORE importing cv2.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
from ultralytics import YOLO


# ------------------------------------------------------------------ thread 1: reader
def reader_loop(rtsp_url, frame_slot, stats, stop_event, retries=10, delay=1.0):
    cap = None
    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            break
        cap.release()
        print(f"[WAIT] Stream not ready ({attempt}/{retries}), retrying in {delay}s...")
        time.sleep(delay)
    else:
        print(f"[ERROR] Cannot open stream: {rtsp_url}")
        stop_event.set()
        return

    failures = 0
    while not stop_event.is_set():
        t0 = time.perf_counter()
        ret, frame = cap.read()
        read_ms = (time.perf_counter() - t0) * 1000

        if not ret:
            failures += 1
            if failures >= 50:
                print("[WARN] Stream lost, reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                failures = 0
            continue

        failures = 0
        stats.add("read", read_ms)
        frame_slot.put((frame, time.perf_counter()))  # frame + capture time

    cap.release()


# ------------------------------------------------------------------ thread 2: inference
def inference_loop(model, frame_slot, result_slot, stats, stop_event, imgsz, conf):
    last_version = 0

    while not stop_event.is_set():
        data, version = frame_slot.get()
        if data is None or version == last_version:
            time.sleep(0.001)  # no new frame yet
            continue
        last_version = version
        frame, capture_time = data

        t0 = time.perf_counter()
        results = model.track(
            frame,
            persist=True,  # keep track IDs between frames
            classes=TARGET_CLASSES,
            imgsz=imgsz,
            conf=conf,
            verbose=False,
        )
        stats.add("inference", (time.perf_counter() - t0) * 1000)

        result = results[0]
        for key, ms in result.speed.items():  # YOLO's own breakdown (ms)
            stats.add(f"yolo_{key}", ms)

        result_slot.put((result, capture_time))


# ------------------------------------------------------------------ latency table
def draw_latency_table(image, means, fps):
    lines = [f"Display FPS: {fps:.1f}"]
    lines += [f"{s}: {means[s]:.1f} ms" for s in STAGES if s in means]

    line_h = 22
    cv2.rectangle(image, (0, 0), (290, 12 + line_h * len(lines)), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(
            image,
            text,
            (10, 25 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
        )


def run_inference(model, rtsp_url, imgsz=960, conf=0.25):
    stats = LatencyStats()
    frame_slot = SharedSlot()
    result_slot = SharedSlot()
    stop_event = threading.Event()

    reader = threading.Thread(
        target=reader_loop,
        args=(rtsp_url, frame_slot, stats, stop_event),
        daemon=True,
    )
    worker = threading.Thread(
        target=inference_loop,
        args=(model, frame_slot, result_slot, stats, stop_event, imgsz, conf),
        daemon=True,
    )
    reader.start()
    worker.start()

    last_version = 0
    fps_t0, fps_frames, fps = time.perf_counter(), 0, 0.0

    try:
        while not stop_event.is_set():
            data, version = result_slot.get()
            if data is None or version == last_version:
                time.sleep(0.001)  # no new result yet
                continue
            last_version = version
            result, capture_time = data

            # display stage
            t0 = time.perf_counter()
            annotated = result.plot()
            draw_latency_table(annotated, stats.means(), fps)
            cv2.imshow("SeeDrive - Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            now = time.perf_counter()

            stats.add("display", (now - t0) * 1000)
            stats.add("end_to_end", (now - capture_time) * 1000)

            # display FPS (recomputed every second)
            fps_frames += 1
            if now - fps_t0 >= 1.0:
                fps = fps_frames / (now - fps_t0)
                fps_t0, fps_frames = now, 0

            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[STOP] Interrupted.")

    finally:
        stop_event.set()
        reader.join(timeout=2)
        worker.join(timeout=2)
        cv2.destroyAllWindows()
