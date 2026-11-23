import argparse

import cv2
import numpy as np

from configs.config import IMAGES_DIR

ZOOM = 1.5               # display scale, for more precise clicks


def main():
    parser = argparse.ArgumentParser(description="Get pixel coordinates by clicking")
    parser.add_argument("--sequence", default="MVI_20011")
    args = parser.parse_args()

    image = cv2.imread(str(IMAGES_DIR / args.sequence / "img00001.jpg"))
    points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point = (round(x / ZOOM), round(y / ZOOM))       # back to real image pixels
            points.append(point)
            print(f"point {len(points)}: {list(point)}")

    cv2.namedWindow("Pick points")
    cv2.setMouseCallback("Pick points", on_click)

    while True:
        view = image.copy()
        for i, point in enumerate(points):
            cv2.circle(view, point, 4, (0, 0, 255), -1)
            cv2.putText(view, str(i + 1), (point[0] + 6, point[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(points) > 1:
            cv2.polylines(view, [np.int32(points)], isClosed=False, color=(0, 0, 255), thickness=1)
        cv2.imshow("Pick points", cv2.resize(view, None, fx=ZOOM, fy=ZOOM))

        key = cv2.waitKey(20) & 0xFF
        if key == ord("c"):
            print("YAML:", [list(p) for p in points])           # ready to paste into the YAML
        elif key == ord("r"):
            points.clear()
            print("[RESET]")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()