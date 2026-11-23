import argparse
import json

import cv2
import numpy as np

from configs.config import HOMOGRAPHY_DIR, IMAGES_DIR

PX_PER_M = 10  # bird-view resolution: 10 pixels = 1 meter
X_RANGE = (-15, 20)  # meters shown across the road (A is x = 0)
Y_RANGE = (
    -10,
    90,
)  # meters shown along the road (A is y = 0, positive = away from camera)


def load_homography(seq):

    data = json.loads((HOMOGRAPHY_DIR / f"{seq}.json").read_text(encoding="utf-8"))
    return np.array(data["H"], dtype=np.float64)


def bird_view(image, H):
    """Warp the camera image onto the ground plane, with a grid every 5 m."""
    xmin, xmax = X_RANGE
    ymin, ymax = Y_RANGE
    s = PX_PER_M

    # meters -> bird-view pixels (y flipped so that far from the camera = top of the picture)
    to_pixels = np.array(
        [[s, 0, -xmin * s], [0, -s, ymax * s], [0, 0, 1]], dtype=np.float64
    )
    size = ((xmax - xmin) * s, (ymax - ymin) * s)
    view = cv2.warpPerspective(
        image, to_pixels @ H, size
    )  # image pixels -> meters -> bird pixels

    for x in range(xmin, xmax + 1, 5):
        px = (x - xmin) * s
        cv2.line(view, (px, 0), (px, size[1]), (0, 255, 255), 1)
    for y in range(ymin, ymax + 1, 5):
        py = (ymax - y) * s
        cv2.line(view, (0, py), (size[0], py), (0, 255, 255), 1)
        cv2.putText(
            view, f"{y} m", (3, py - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
        )
    return view


def main():
    parser = argparse.ArgumentParser(
        description="Bird's-eye view to check a homography"
    )
    parser.add_argument("--sequence", default="MVI_20011")
    args = parser.parse_args()

    image = cv2.imread(str(IMAGES_DIR / args.sequence / "img00001.jpg"))
    cv2.imshow("Camera", image)
    cv2.imshow("Bird's-eye view", bird_view(image, load_homography(args.sequence)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
