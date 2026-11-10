"""
logic/show_speeds.py
Replay a sequence and draw each tracked vehicle with its number and real speed.
Keys: SPACE = pause / resume, q = quit

Run from the project root:
    python -m logic.show_speeds --sequence MVI_20011
"""

import argparse

import cv2

from configs.config import IMAGES_DIR, TRACKERS_DIR
from logic.check_homography import load_homography
from logic.speed import load_tracks, speeds_per_frame

STOPPED_KMH = 5        # below this, the remaining noise is shown as "stopped"


def draw_label(image, text, x, y):
    """White text with a black outline: readable on any background."""
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description="Display vehicle speeds on a sequence")
    parser.add_argument("--sequence", default="MVI_20011")
    parser.add_argument("--tracker", default="SORT")
    parser.add_argument("--delay", type=int, default=40, help="ms between frames (40 = 25 fps)")
    args = parser.parse_args()

    H = load_homography(args.sequence)
    tracks = load_tracks(TRACKERS_DIR / args.tracker / "data" / f"{args.sequence}.txt")
    per_frame = speeds_per_frame(tracks, H)

    images = sorted((IMAGES_DIR / args.sequence).glob("img*.jpg"))
    paused = False
    for image_path in images:
        frame = int(image_path.stem[3:])                   # "img00012" -> 12
        image = cv2.imread(str(image_path))

        for track_id, (x1, y1, x2, y2), speed in per_frame.get(frame, []):
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
            if speed is None:
                label = f"#{track_id}"                     # too far, or just appeared
            elif speed < STOPPED_KMH:
                label = f"#{track_id} stopped"
            else:
                label = f"#{track_id} {speed:.0f} km/h"
            draw_label(image, label, int(x1), int(y1) - 5)

        draw_label(image, f"{args.sequence} | frame {frame}", 10, 25)
        cv2.imshow("SeeDrive - speeds", image)

        key = cv2.waitKey(0 if paused else args.delay) & 0xFF
        if key == ord(" "):
            paused = not paused
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()