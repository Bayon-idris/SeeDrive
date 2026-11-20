"""
logic/engine.py
Everything SeeDrive knows and decides for ONE camera, frame by frame:
tracked vehicles -> speed and direction -> rules -> alerts -> drawing.
Used by the replay demo (logic/replay.py) and later by the live pipeline.
"""

import cv2

from configs.config import FPS
from logic.alerts import AlertLog
from logic.check_homography import load_homography
from logic.rules import ForbiddenZoneDetector, LineCounter, StoppedDetector, WrongWayDetector
from logic.scene import draw_scene, load_scene
from logic.vehicles import Vehicles

SWAP = {"toward_camera": "away_from_camera", "away_from_camera": "toward_camera"}


def draw_label(image, text, x, y, scale=0.5):
    """White text with a black outline: readable on any background."""
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)


class TrafficLogic:
    def __init__(self, camera_id, simulate=False):
        self.scene = load_scene(camera_id)
        t = self.scene["thresholds"]
        directions = [z for z in self.scene["zones"] if z["type"] == "direction"]
        forbidden = [z for z in self.scene["zones"] if z["type"] == "forbidden"]
        if simulate:                               # demo mode: swap expected directions -> wrong-way alerts
            directions = [{**z, "expected": SWAP[z["expected"]]} for z in directions]

        self.vehicles = Vehicles(load_homography(self.scene["sequence"]))
        self.counter = LineCounter(self.scene["counting_line"])
        self.stopped = StoppedDetector([z["polygon"] for z in directions],
                                       t["stopped_kmh"], t["stopped_seconds"])
        self.wrong_way = WrongWayDetector(directions, t["wrong_way_kmh"], t["wrong_way_seconds"])
        self.intrusions = ForbiddenZoneDetector(forbidden, t["forbidden_seconds"])
        self.alerts = AlertLog(camera_id)
        self.visible = []                          # vehicles seen on the current frame
        self.boxes = {}                            # their boxes, for drawing

    def update(self, frame, tracks):
        """tracks: [(track_id, (x1, y1, x2, y2)), ...] given by the tracker for this frame."""
        self.visible = self.vehicles.update(frame, tracks)
        self.boxes = {track_id: box for track_id, box in tracks}

        self.counter.update(self.visible)
        for v in self.stopped.update(frame, self.visible):
            self.alerts.add(frame, "stopped", v, f"more than {self.stopped.min_frames // FPS} s")
        for v in self.wrong_way.update(frame, self.visible):
            self.alerts.add(frame, "wrong_way", v, v.direction)
        for v, zone_name in self.intrusions.update(frame, self.visible):
            self.alerts.add(frame, "forbidden_zone", v, zone_name)

    def draw(self, image, frame):
        """Scene + vehicles (box, number, speed) + counts + alert banners."""
        draw_scene(image, self.scene)
        for v in self.visible:
            x1, y1, x2, y2 = map(int, self.boxes[v.id])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
            speed = f" {v.speed:.0f} km/h" if v.speed is not None else ""
            draw_label(image, f"#{v.id}{speed}", x1, y1 - 5)

        c = self.counter.counts
        draw_label(image, f"Toward camera: {c['toward_camera']}   Away: {c['away_from_camera']}",
                   10, 25, scale=0.7)
        self.alerts.draw(image, frame)
        return image