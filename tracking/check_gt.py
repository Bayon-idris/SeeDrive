"""
tracking/check_gt.py
Step 2 test: draw the ground truth boxes WITH their IDs on the images.
Each vehicle must keep the same ID (and the same color) from start to end.
Ignored regions are shown in gray.

"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2

from configs.config import GT_DIR, IGNORED_DIR, IMAGE_EXTENSIONS, IMAGES_DIR


def frame_number(image_path):
    digits = "".join(c for c in image_path.stem if c.isdigit())
    return int(digits)


def id_color(track_id):
    """Same ID -> same color, different IDs -> different colors."""
    return ((track_id * 37) % 256, (track_id * 91) % 256, (track_id * 151) % 256)


def load_gt(gt_path):
    """Returns {frame: [(id, x, y, w, h), ...]}"""
    gt = defaultdict(list)
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame, track_id = int(parts[0]), int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            gt[frame].append((track_id, x, y, w, h))
    return gt


def load_ignored(path):
    regions = []
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    regions.append(tuple(map(float, line.strip().split(","))))
    return regions


def main():
    parser = argparse.ArgumentParser(description="Display ground truth with IDs")
    parser.add_argument(
        "--file", default=None, help="MOT file to display (default: ground truth)"
    )
    parser.add_argument("--sequence", default="MVI_20011")
    parser.add_argument("--delay", type=int, default=40)
    args = parser.parse_args()

    gt_path = Path(args.file) if args.file else GT_DIR / args.sequence / "gt" / "gt.txt"
    if not gt_path.is_file():
        print(f"[ERROR] Ground truth not found: {gt_path}")
        return

    gt = load_gt(gt_path)
    ignored = load_ignored(IGNORED_DIR / f"{args.sequence}.txt")
    images = sorted(
        (
            p
            for p in (IMAGES_DIR / args.sequence).iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=frame_number,
    )

    paused = False
    for image_path in images:
        frame_id = frame_number(image_path)
        image = cv2.imread(str(image_path))

        # ignored regions: gray, semi-transparent
        overlay = image.copy()
        for x, y, w, h in ignored:
            cv2.rectangle(
                overlay, (int(x), int(y)), (int(x + w), int(y + h)), (128, 128, 128), -1
            )
        image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)

        # ground truth boxes with IDs
        for track_id, x, y, w, h in gt.get(frame_id, []):
            color = id_color(track_id)
            p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
            cv2.rectangle(image, p1, p2, color, 2)
            cv2.putText(
                image,
                f"id:{track_id}",
                (p1[0], p1[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        info = (
            f"{args.sequence} | frame {frame_id} | {len(gt.get(frame_id, []))} vehicles"
        )
        cv2.putText(
            image, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.imshow("Check ground truth", image)

        key = cv2.waitKey(0 if paused else args.delay) & 0xFF
        if key == ord(" "):
            paused = not paused
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
