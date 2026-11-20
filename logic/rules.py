"""
logic/rules.py
Traffic rules applied to the live vehicles of one camera.

Run from the project root (test on the recorded SORT results of cam_01):
    python -m logic.rules
    python -m logic.rules --simulate
"""

import cv2
import numpy as np

from configs.config import FPS


# ------------------------------------------------------------------ geometry helpers
def cross(o, a, b):
    """Cross product of OA and OB: > 0 if B is on one side of line OA, < 0 on the other."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(p1, p2, q1, q2):
    """True if segment p1-p2 crosses segment q1-q2."""
    return (cross(q1, q2, p1) * cross(q1, q2, p2) < 0 and     # p1, p2 on both sides of q
            cross(p1, p2, q1) * cross(p1, p2, q2) < 0)        # q1, q2 on both sides of p


def inside(point, polygon):
    """True if a pixel point is inside (or on the edge of) a polygon."""
    return cv2.pointPolygonTest(np.float32(polygon), (float(point[0]), float(point[1])), False) >= 0


# ------------------------------------------------------------------ rule 1: counting
class LineCounter:
    """Counts vehicles crossing the counting line, per direction."""

    def __init__(self, line):
        self.a, self.b = [tuple(p) for p in line]
        self.counts = {"toward_camera": 0, "away_from_camera": 0}
        self.counted = set()                      # ids already counted: count each vehicle once

    def update(self, vehicles):
        """Check the vehicles visible on this frame. Returns [(vehicle, direction), ...] crossing now."""
        crossings = []
        for v in vehicles:
            if v.prev_foot is None or v.id in self.counted:
                continue
            if segments_intersect(v.prev_foot, v.foot, self.a, self.b):   # moved across the line
                direction = "toward_camera" if v.foot[1] > v.prev_foot[1] else "away_from_camera"
                self.counts[direction] += 1
                self.counted.add(v.id)
                crossings.append((v, direction))
        return crossings


# ------------------------------------------------------------------ rule 2: stopped vehicle
class StoppedDetector:
    """Alert when a vehicle stays (almost) still on a road for too long."""

    def __init__(self, road_polygons, stopped_kmh, stopped_seconds):
        self.roads = road_polygons
        self.stopped_kmh = stopped_kmh
        self.min_frames = int(stopped_seconds * FPS)
        self.slow_since = {}          # id -> first frame of the current "slow" period
        self.alerted = set()          # ids already reported: one alert per vehicle

    def update(self, frame, vehicles):
        """Returns the vehicles that just became 'stopped' on this frame."""
        alerts = []
        for v in vehicles:
            on_road = any(inside(v.foot, road) for road in self.roads)
            if not on_road or v.speed is None or v.speed >= self.stopped_kmh:
                self.slow_since.pop(v.id, None)          # moving again (or off the road): reset
                continue
            start = self.slow_since.setdefault(v.id, frame)
            if frame - start >= self.min_frames and v.id not in self.alerted:
                self.alerted.add(v.id)
                alerts.append(v)
        return alerts


# ------------------------------------------------------------------ rule 3: wrong way
class WrongWayDetector:
    """Alert when a vehicle drives against the expected direction of its road."""

    def __init__(self, direction_zones, wrong_way_kmh, wrong_way_seconds):
        self.zones = direction_zones              # list of {"polygon": ..., "expected": ...}
        self.min_kmh = wrong_way_kmh
        self.min_frames = int(wrong_way_seconds * FPS)
        self.wrong_since = {}                     # id -> first frame of the current wrong-way period
        self.alerted = set()

    def update(self, frame, vehicles):
        """Returns the vehicles that just started driving the wrong way."""
        alerts = []
        for v in vehicles:
            zone = next((z for z in self.zones if inside(v.foot, z["polygon"])), None)
            wrong = (zone is not None and v.direction is not None
                     and v.speed >= self.min_kmh and v.direction != zone["expected"])
            if not wrong:
                self.wrong_since.pop(v.id, None)
                continue
            start = self.wrong_since.setdefault(v.id, frame)
            if frame - start >= self.min_frames and v.id not in self.alerted:
                self.alerted.add(v.id)
                alerts.append(v)
        return alerts
    
# ------------------------------------------------------------------ rule 4: forbidden zone
class ForbiddenZoneDetector:
    """Alert when a vehicle stays inside a forbidden zone for too long."""

    def __init__(self, forbidden_zones, forbidden_seconds):
        self.zones = forbidden_zones              # list of {"name": ..., "polygon": ...}
        self.min_frames = int(forbidden_seconds * FPS)
        self.inside_since = {}                    # (id, zone name) -> first frame inside
        self.alerted = set()                      # (id, zone name) already reported

    def update(self, frame, vehicles):
        """Returns [(vehicle, zone name), ...] for new intrusions on this frame."""
        alerts = []
        for v in vehicles:
            for zone in self.zones:
                key = (v.id, zone["name"])
                if not inside(v.foot, zone["polygon"]):
                    self.inside_since.pop(key, None)      # left the zone: reset
                    continue
                start = self.inside_since.setdefault(key, frame)
                if frame - start >= self.min_frames and key not in self.alerted:
                    self.alerted.add(key)
                    alerts.append((v, zone["name"]))
        return alerts    

# ------------------------------------------------------------------ test
if __name__ == "__main__":
    import argparse
    from collections import defaultdict

    from configs.config import TRACKERS_DIR
    from logic.check_homography import load_homography
    from logic.scene import load_scene
    from logic.speed import load_tracks
    from logic.vehicles import Vehicles

    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true",
                        help="test mode: every vehicle on a road counts as 'stopped' after 2 s")
    args = parser.parse_args()

    scene = load_scene("cam_01")
    t = scene["thresholds"]
    roads = [z["polygon"] for z in scene["zones"] if z["type"] == "direction"]
    directions = [z for z in scene["zones"] if z["type"] == "direction"]
    if args.simulate:                                   # test mode: swap the expected directions
        swap = {"toward_camera": "away_from_camera", "away_from_camera": "toward_camera"}
        directions = [{**z, "expected": swap[z["expected"]]} for z in directions]
    
    forbidden = [z for z in scene["zones"] if z["type"] == "forbidden"]
    if args.simulate:                       # test mode: the main road also becomes forbidden
        forbidden += [{**z, "name": z["name"] + "_test"}
                      for z in scene["zones"] if z["name"] == "main_road"]    

    tracks = load_tracks(TRACKERS_DIR / "SORT" / "data" / f"{scene['sequence']}.txt")
    per_frame = defaultdict(list)                 # simulate the live tracker, frame by frame
    for track_id, rows in tracks.items():
        for frame, *box in rows:
            per_frame[frame].append((track_id, box))

    vehicles = Vehicles(load_homography(scene["sequence"]))
    counter = LineCounter(scene["counting_line"])
    if args.simulate:
        stopped = StoppedDetector(roads, stopped_kmh=1000, stopped_seconds=2)
    else:
        stopped = StoppedDetector(roads, t["stopped_kmh"], t["stopped_seconds"])
    wrong_way = WrongWayDetector(directions, t["wrong_way_kmh"], t["wrong_way_seconds"])
    intrusions = ForbiddenZoneDetector(forbidden, t["forbidden_seconds"])

    for frame in range(1, max(per_frame) + 1):
        visible = vehicles.update(frame, per_frame.get(frame, []))
        counter.update(visible)
        for v in stopped.update(frame, visible):
            print(f"frame {frame:4d}: ALERT stopped vehicle #{v.id} ({v.speed:.0f} km/h)")
        for v in wrong_way.update(frame, visible):
            print(f"frame {frame:4d}: ALERT wrong way #{v.id} ({v.direction}, {v.speed:.0f} km/h)")    
        for v, zone_name in intrusions.update(frame, visible):
            print(f"frame {frame:4d}: ALERT forbidden zone '{zone_name}' #{v.id}")    
    print("\nCounts:", counter.counts)
    print("Stopped vehicles:", len(stopped.alerted))
    print("Wrong-way vehicles:", len(wrong_way.alerted))
    print("Forbidden zone intrusions:", len(intrusions.alerted))