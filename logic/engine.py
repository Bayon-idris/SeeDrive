from configs.config import FPS
from logic.alerts import AlertLog
from logic.check_homography import load_homography
from logic.overlay import draw_label, draw_lines, draw_panel, draw_vehicle, route_rows
from logic.rules import (ForbiddenZoneDetector, LineCounter, RouteCounter, StoppedDetector,
                         WrongWayDetector)
from logic.scene import draw_scene, load_scene
from logic.vehicles import Vehicles

SWAP = {"toward_camera": "away_from_camera", "away_from_camera": "toward_camera"}


class TrafficLogic:
    """Every rule of one camera. Each part is optional: it runs only if the camera's
    YAML defines it (homography -> speed, lines -> routes, counting_line, zones -> alerts)."""

    def __init__(self, camera_id, simulate=False, class_names=None):
        self.scene = load_scene(camera_id)
        self.class_names = list(class_names or ["vehicle"])
        H = load_homography(self.scene["sequence"]) if self.scene.get("homography") else None
        self.vehicles = Vehicles(H)

        lines = self.scene.get("lines", [])
        self.routes = RouteCounter(lines, self.scene.get("routes")) if lines else None
        line = self.scene.get("counting_line")
        self.counter = LineCounter(line) if line else None

        zones = self.scene.get("zones", [])
        directions = [z for z in zones if z["type"] == "direction"]
        forbidden = [z for z in zones if z["type"] == "forbidden"]
        if simulate:                     # demo mode: swap expected directions -> wrong-way alerts
            directions = [{**z, "expected": SWAP[z["expected"]]} for z in directions]
        t = self.scene.get("thresholds", {})
        self.rules = []
        if directions and H is not None:          # these two rules need speeds
            self.rules.append(("stopped", StoppedDetector([z["polygon"] for z in directions],
                                                          t["stopped_kmh"], t["stopped_seconds"])))
            self.rules.append(("wrong_way", WrongWayDetector(directions, t["wrong_way_kmh"],
                                                             t["wrong_way_seconds"])))
        if forbidden:
            self.rules.append(("forbidden_zone", ForbiddenZoneDetector(forbidden,
                                                                       t["forbidden_seconds"])))
        self.alerts = AlertLog(camera_id)
        self.visible = []                # vehicles seen on the current frame
        self.boxes = {}                  # their boxes, for drawing

    def update(self, frame, tracks):
        """tracks: [(track_id, (x1, y1, x2, y2), class_name, conf), ...] for this frame."""
        self.visible = self.vehicles.update(frame, tracks)
        self.boxes = {t[0]: t[1] for t in tracks}

        if self.routes:
            self.routes.update(frame, self.visible)
        if self.counter:
            self.counter.update(self.visible)
        for kind, rule in self.rules:
            for hit in rule.update(frame, self.visible):
                if kind == "forbidden_zone":
                    v, details = hit                        # (vehicle, zone name)
                elif kind == "stopped":
                    v, details = hit, f"more than {rule.min_frames // FPS} s"
                else:
                    v, details = hit, hit.direction
                self.alerts.add(frame, kind, v, details)

    def draw(self, image, frame):
        """Scene + route lines + vehicles + counts panel + alert banners."""
        draw_scene(image, self.scene)
        if self.routes:
            draw_lines(image, self.scene["lines"], self.routes.last_crossed, frame)
        for v in self.visible:
            draw_vehicle(image, v, self.boxes[v.id])

        if self.routes:
            draw_panel(image, route_rows(self.routes.counts, self.class_names))
        if self.counter:
            c = self.counter.counts
            draw_label(image, f"Toward camera: {c['toward_camera']}   Away: {c['away_from_camera']}",
                       10, 25, scale=0.7)
        self.alerts.draw(image, frame)
        return image