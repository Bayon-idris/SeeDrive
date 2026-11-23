import os
import threading
import time

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp"
)  # before importing cv2

import cv2
import numpy as np
from ultralytics import YOLO

from configs.config import CONFIG_PATH, FPS, MODEL_PATH, OUTPUTS_DIR, ROOT
import sort_tracker  # noqa: E402  (configs.config adds the C++ build folder to sys.path)
from ingest.rtsp_reader import load_cameras, start_cameras, stop_cameras
from logic.engine import TrafficLogic
from logic.overlay import draw_tag


class Latest:
    """A box with one place, shared by two threads (same idea as SharedSlot)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = None
        self.version = 0  # increases at each put(): tells the reader it's new

    def put(self, data):
        with self.lock:
            self.data = data
            self.version += 1

    def get(self):
        with self.lock:
            return self.data, self.version


def iou_matrix(a, b):
    """IoU of every box of a (M, 4) with every box of b (N, 4) -> (M, N)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def add_classes(tracks, dets, classes, names, min_iou=0.3):
    """Our C++ SORT only returns boxes + IDs. Give each track the class and confidence
    of the detection it overlaps most. -> [(id, box, class_name, conf), ...]"""
    if len(tracks) == 0:
        return []
    ious = (
        iou_matrix(tracks[:, :4], dets[:, :4])
        if len(dets)
        else np.zeros((len(tracks), 0))
    )
    out = []
    for i, t in enumerate(tracks):
        j = int(ious[i].argmax()) if ious.shape[1] else -1
        if j >= 0 and ious[i, j] >= min_iou:
            out.append(
                (
                    int(t[4]),
                    tuple(map(float, t[:4])),
                    names[int(classes[j])],
                    float(dets[j, 4]),
                )
            )
        else:  # predicted box, no detection this frame
            out.append((int(t[4]), tuple(map(float, t[:4])), None, None))
    return out


# ------------------------------------------------------------------ thread 1: reader
def reader_loop(url, frames, stop):
    for _ in range(10):  # the publisher may need a few seconds to start
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
        frame_no += 1  # counts EVERY frame read: real time = frame_no / FPS
        frames.put((frame_no, image))
    cap.release()


# ------------------------------------------------------------------ thread 2: perception
def perception_loop(model, frames, results, stop):
    tracker = sort_tracker.Sort(
        max_age=30, min_hits=3, iou_threshold=0.3, det_threshold=0.5
    )
    last_version = 0
    while not stop.is_set():
        data, version = frames.get()
        if data is None or version == last_version:
            time.sleep(0.001)  # no new frame yet
            continue
        last_version = version
        frame_no, image = data

        result = model.predict(image, imgsz=960, conf=0.1, verbose=False)[
            0
        ]  # detection
        dets = np.hstack(
            [
                result.boxes.xyxy.cpu().numpy(),  # (N, 5):
                result.boxes.conf.cpu().numpy()[:, None],
            ]
        )  # x1 y1 x2 y2 score
        classes = result.boxes.cls.cpu().numpy()
        tracks = add_classes(
            tracker.update(dets), dets, classes, model.names
        )  # C++ SORT: + IDs
        results.put((frame_no, image, tracks))


# ------------------------------------------------------------------ main thread
def run(camera_id, simulate=False, save=False):
    """Run the full live pipeline on one camera (until 'q' or Ctrl+C).
    save=True also records what is shown in outputs/demo_<camera>.mp4"""
    camera = next((c for c in load_cameras(CONFIG_PATH) if c["id"] == camera_id), None)
    if camera is None:
        print(f"[ERROR] {camera_id} not found or not enabled in {CONFIG_PATH}")
        return

    model = YOLO(str(MODEL_PATH))
    names = [model.names[k] for k in sorted(model.names)]
    logic = TrafficLogic(camera_id, simulate=simulate, class_names=names)
    writer = None
    frames, results, stop = Latest(), Latest(), threading.Event()

    processes = start_cameras([camera])  # FFmpeg publishes the simulated camera
    threads = [
        threading.Thread(
            target=reader_loop, args=(camera["rtsp_url"], frames, stop), daemon=True
        ),
        threading.Thread(
            target=perception_loop, args=(model, frames, results, stop), daemon=True
        ),
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

            logic.update(frame_no, tracks)  # speeds, counts, rules, alerts
            logic.draw(image, frame_no)

            fps_frames += 1  # processed frames per second
            now = time.perf_counter()
            if now - fps_t0 >= 1.0:
                fps, fps_frames, fps_t0 = fps_frames / (now - fps_t0), 0, now
            draw_tag(
                image,
                f"{camera_id} | {fps:.1f} FPS",
                image.shape[1] - 175,
                28,
                (40, 40, 40),
                (255, 255, 255),
                0.55,
            )

            if save:
                if writer is None:
                    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
                    out_path = OUTPUTS_DIR / f"demo_{camera_id}.mp4"
                    h, w = image.shape[:2]
                    writer = cv2.VideoWriter(
                        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h)
                    )
                writer.write(image)

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
        if writer is not None:
            writer.release()
            print(f"[OK] Demo video saved -> {out_path}")
        if logic.counter:
            print("Counts:", logic.counter.counts)
        if logic.routes:
            for (origin, dest), counts in logic.routes.counts.items():
                print(f"{origin} -> {dest}: {dict(counts)}")
