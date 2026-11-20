"""
logic/alerts.py
Alerts of one camera: printed, written to a log file (one JSON per line),
and drawn as red banners on the video for a few seconds.
"""

import json
from collections import deque
from datetime import datetime

import cv2

from configs.config import FPS, OUTPUTS_DIR

ALERTS_DIR = OUTPUTS_DIR / "alerts"
SHOW_SECONDS = 3            # how long an alert stays on screen


class AlertLog:
    def __init__(self, camera_id):
        self.camera = camera_id
        ALERTS_DIR.mkdir(parents=True, exist_ok=True)
        self.path = ALERTS_DIR / f"{camera_id}.jsonl"
        self.recent = deque(maxlen=5)          # last alerts, for the on-screen banners

    def add(self, frame, kind, vehicle, details=""):
        """Record one alert: print it, append it to the log file, keep it for display."""
        alert = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "camera": self.camera,
            "frame": frame,
            "type": kind,                                   # stopped / wrong_way / forbidden_zone
            "vehicle_id": int(vehicle.id),
            "speed_kmh": round(vehicle.speed, 1) if vehicle.speed is not None else None,
            "details": details,
        }
        with open(self.path, "a", encoding="utf-8") as f:   # "a" = append, never overwrite
            f.write(json.dumps(alert) + "\n")
        print(f"[ALERT] frame {frame}: {kind} #{vehicle.id} {details}")
        self.recent.append(alert)

    def draw(self, image, frame):
        """Red banners at the bottom of the image for the alerts of the last few seconds."""
        active = [a for a in self.recent if frame - a["frame"] <= SHOW_SECONDS * FPS]
        for i, a in enumerate(reversed(active)):            # newest at the bottom
            text = f"{a['type'].upper()}  vehicle #{a['vehicle_id']}  {a['details']}"
            y = image.shape[0] - 15 - 30 * i
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (5, y - h - 8), (w + 20, y + 8), (0, 0, 200), -1)
            cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return image


if __name__ == "__main__":
    from types import SimpleNamespace

    import numpy as np

    log = AlertLog("cam_test")
    log.add(10, "wrong_way", SimpleNamespace(id=7, speed=95.3), "toward_camera")
    log.add(12, "stopped", SimpleNamespace(id=9, speed=1.2), "12 s")

    image = log.draw(np.full((540, 960, 3), 90, np.uint8), frame=20)
    cv2.imshow("Alerts test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("\nLog file content:")
    print(log.path.read_text(encoding="utf-8"))