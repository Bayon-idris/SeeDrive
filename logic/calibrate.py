import argparse
import json

import cv2
import numpy as np

from configs.config import HOMOGRAPHY_DIR, IMAGES_DIR
from logic.check_homography import bird_view

LABELS = ["A", "B", "C", "D"]
ZOOM = 1.5  # display scale, for more precise clicks


def compute_homography(image_points, width, length):

    world_points = [(0, 0), (width, 0), (width, length), (0, length)]
    return cv2.getPerspectiveTransform(
        np.float32(image_points), np.float32(world_points)
    )


def main():
    parser = argparse.ArgumentParser(description="Click 4 points to calibrate a camera")
    parser.add_argument("--sequence", default="MVI_20011")
    parser.add_argument("--width", type=float, default=3.5, help="lane width (m)")
    parser.add_argument(
        "--length", type=float, default=15.0, help="one dash + one gap (m)"
    )
    args = parser.parse_args()

    image = cv2.imread(str(IMAGES_DIR / args.sequence / "img00001.jpg"))
    clicks = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((round(x / ZOOM), round(y / ZOOM)))

    cv2.namedWindow("Calibrate")
    cv2.setMouseCallback(
        "Calibrate", on_click
    )  # OpenCV calls on_click at each mouse event

    while True:
        view = image.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(view, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(
                view,
                LABELS[i],
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        if len(clicks) == 4:
            cv2.polylines(
                view, [np.int32(clicks)], isClosed=True, color=(0, 0, 255), thickness=1
            )
            H = compute_homography(clicks, args.width, args.length)
            cv2.imshow("Bird's-eye view", bird_view(image, H))
        cv2.imshow("Calibrate", cv2.resize(view, None, fx=ZOOM, fy=ZOOM))

        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            clicks.clear()
        elif key == ord("q"):
            break
        elif key == ord("s") and len(clicks) == 4:
            HOMOGRAPHY_DIR.mkdir(parents=True, exist_ok=True)
            out_path = HOMOGRAPHY_DIR / f"{args.sequence}.json"
            out_path.write_text(
                json.dumps(
                    {
                        "sequence": args.sequence,
                        "image_points": clicks,
                        "width": args.width,
                        "length": args.length,
                        "H": H.tolist(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[OK] Homography saved -> {out_path}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
