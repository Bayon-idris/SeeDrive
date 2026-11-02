"""
tracking/check_detections.py
Step 1 test: draw the saved detections on the original images.

    Green box  : confidence >= 0.5
    Orange box : confidence < 0.5 (low-confidence detections)

Keys: SPACE = pause / resume, q = quit
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "DETRAC-Images" / "DETRAC-Images"
DETECTIONS_DIR = ROOT / "outputs" / "detections"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def frame_number(image_path):
    digits = "".join(c for c in image_path.stem if c.isdigit())
    return int(digits)


def load_detections(txt_path):
    """Returns {frame: [(x, y, w, h, conf), ...]}"""
    detections = defaultdict(list)
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame = int(parts[0])
            x, y, w, h, conf = map(float, parts[2:7])
            detections[frame].append((x, y, w, h, conf))
    return detections


def main():
    parser = argparse.ArgumentParser(description="Display saved detections")
    parser.add_argument("--sequence", default="MVI_20011")
    parser.add_argument("--delay", type=int, default=40, help="ms between frames (40 = 25 fps)")
    args = parser.parse_args()

    seq_dir = DATASET_DIR / args.sequence
    txt_path = DETECTIONS_DIR / f"{args.sequence}.txt"

    if not txt_path.is_file():
        print(f"[ERROR] Detections file not found: {txt_path}")
        return

    detections = load_detections(txt_path)
    images = sorted(
        (p for p in seq_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
        key=frame_number,
    )
    print(f"[INFO] {len(images)} frames, {sum(len(v) for v in detections.values())} boxes loaded")

    paused = False
    for image_path in images:
        frame_id = frame_number(image_path)
        image = cv2.imread(str(image_path))

        for x, y, w, h, conf in detections.get(frame_id, []):
            color = (0, 200, 0) if conf >= 0.5 else (0, 165, 255)
            p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
            cv2.rectangle(image, p1, p2, color, 2)
            cv2.putText(image, f"{conf:.2f}", (p1[0], p1[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        info = f"{args.sequence} | frame {frame_id} | {len(detections.get(frame_id, []))} boxes"
        cv2.putText(image, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Check detections", image)

        key = cv2.waitKey(0 if paused else args.delay) & 0xFF
        if key == ord(" "):
            paused = not paused
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()