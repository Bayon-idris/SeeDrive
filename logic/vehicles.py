from collections import Counter, deque
from math import hypot

from configs.config import FPS
from logic.speed import MAX_DISTANCE, SMOOTH_WINDOW, SPEED_WINDOW, ground_point

MOVING_KMH = 5


class Vehicle:
    """Everything we know about ONE tracked vehicle."""

    def __init__(self, track_id):
        self.id = track_id
        self.history = deque(maxlen=2 * SPEED_WINDOW)  # last (frame, X, Y) on the ground, max 2 s
        self.recent_speeds = deque(maxlen=SMOOTH_WINDOW)
        self.speed = None        # smoothed km/h, None until we have 0.5 s of history (or no homography)
        self.vy = 0.0            # m/s along the road: < 0 = toward the camera
        self.foot = None         # bottom-center of the box, in pixels (for zones)
        self.prev_foot = None    # same point on the previous frame
        self.center = None       # center of the box, in pixels (for the route lines)
        self.prev_center = None
        self.last_frame = None
        self.class_votes = Counter()   # class name -> sum of confidences seen for this vehicle
        self.conf = None               # confidence of the last matched detection

    def update(self, frame, box, H, cls=None, conf=None):
        x1, y1, x2, y2 = box
        self.prev_foot, self.foot = self.foot, ((x1 + x2) / 2, y2)
        self.prev_center, self.center = self.center, ((x1 + x2) / 2, (y1 + y2) / 2)
        self.last_frame = frame
        if cls is not None:                    # the detector's class can change between frames:
            self.class_votes[cls] += conf      # keep a vote, the label is the winner
            self.conf = conf

        if H is None:                          # no calibration for this camera: no speed
            return
        X, Y = ground_point(box, H)
        if Y > MAX_DISTANCE:                   # too far: position too imprecise
            return
        self.history.append((frame, X, Y))

        # oldest position at most 1 s ago (history is ordered, oldest first)
        f0, X0, Y0 = next(h for h in self.history if frame - h[0] <= SPEED_WINDOW)
        if frame - f0 < SPEED_WINDOW // 2:     # less than 0.5 s of history: wait
            return
        seconds = (frame - f0) / FPS
        self.recent_speeds.append(hypot(X - X0, Y - Y0) / seconds * 3.6)
        self.speed = sum(self.recent_speeds) / len(self.recent_speeds)
        self.vy = (Y - Y0) / seconds

    @property
    def label(self):
        """Most voted class name ('vehicle' if the detector gave no class)."""
        return self.class_votes.most_common(1)[0][0] if self.class_votes else "vehicle"

    @property
    def direction(self):
        """'toward_camera', 'away_from_camera', or None if (almost) stopped."""
        if self.speed is None or self.speed < MOVING_KMH:
            return None
        return "toward_camera" if self.vy < 0 else "away_from_camera"


class Vehicles:
    """All vehicles currently tracked on one camera."""

    def __init__(self, H=None, forget_after=2 * FPS):
        self.H = H                          # None = camera not calibrated
        self.forget_after = forget_after    # frames without being seen before we forget a vehicle
        self.by_id = {}

    def update(self, frame, tracks):
        """tracks: list of (track_id, (x1, y1, x2, y2)) or (track_id, box, class_name, conf).
        Returns the Vehicle objects visible on this frame."""
        for track_id, box, *extra in tracks:
            if track_id not in self.by_id:
                self.by_id[track_id] = Vehicle(track_id)
            self.by_id[track_id].update(frame, box, self.H, *extra)

        self.by_id = {tid: v for tid, v in self.by_id.items()          # forget lost vehicles
                      if frame - v.last_frame <= self.forget_after}
        return [self.by_id[t[0]] for t in tracks]


if __name__ == "__main__":
    from collections import defaultdict

    from configs.config import TRACKERS_DIR
    from logic.check_homography import load_homography
    from logic.speed import load_tracks

    H = load_homography("MVI_20011")
    tracks = load_tracks(TRACKERS_DIR / "SORT" / "data" / "MVI_20011.txt")

    per_frame = defaultdict(
        list
    )  # simulate the live tracker: what it gives at each frame
    for track_id, rows in tracks.items():
        for frame, *box in rows:
            per_frame[frame].append((track_id, box))

    vehicles = Vehicles(H)
    for frame in range(1, max(per_frame) + 1):
        visible = vehicles.update(frame, per_frame.get(frame, []))
        if frame % 25 == 0:
            v19 = vehicles.by_id.get(19)
            info = (
                f"#19 {v19.speed:.0f} km/h {v19.direction}"
                if v19 and v19.speed
                else "#19 -"
            )
            print(
                f"frame {frame:4d}: {len(visible):2d} visible, {len(vehicles.by_id):2d} in memory | {info}"
            )