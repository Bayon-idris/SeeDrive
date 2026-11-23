import argparse

import cv2
import numpy as np
import yaml
from logic.overlay import draw_lines
from configs.config import IMAGES_DIR, ROOT

SCENES_DIR = ROOT / "configs" / "cameras"

COLORS = {"forbidden": (0, 0, 255), "direction": (255, 160, 0), "line": (0, 255, 255)}


def load_scene(camera_id):
    with open(SCENES_DIR / f"{camera_id}.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def draw_scene(image, scene):
    """Draw the counting line, the zones and the expected directions."""
    overlay = image.copy()
    for zone in scene.get("zones", []):
        pts = np.int32(zone["polygon"])
        color = COLORS[zone["type"]]
        if zone["type"] == "forbidden":
            cv2.fillPoly(overlay, [pts], color)  # filled = forbidden
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

        cx, cy = pts.mean(axis=0).astype(int)  # zone center, for labels
        if zone["type"] == "direction":  # arrow = expected direction
            dy = 50 if zone["expected"] == "toward_camera" else -50
            cv2.arrowedLine(
                image, (cx, cy - dy // 2), (cx, cy + dy // 2), color, 3, tipLength=0.4
            )
        cv2.putText(
            image, zone["name"], (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    if scene.get("zones"):
        image[:] = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)

    if "counting_line" not in scene:
        return image
    p1, p2 = [tuple(p) for p in scene["counting_line"]]
    cv2.line(image, p1, p2, COLORS["line"], 2)
    cv2.putText(
        image,
        "counting line",
        (p1[0], p1[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLORS["line"],
        2,
    )
    return image


def main():
    parser = argparse.ArgumentParser(description="Draw a camera's scene settings")
    parser.add_argument("--camera", default="cam_01")
    args = parser.parse_args()

    scene = load_scene(args.camera)
    image = cv2.imread(str(IMAGES_DIR / scene["sequence"] / "img00001.jpg"))
    draw_scene(image, scene)
    draw_lines(image, scene.get("lines", []), {}, frame=0)
    cv2.imshow("Scene", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
