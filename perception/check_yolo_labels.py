"""
Draw YOLO labels back on random images of the fine-tuning dataset.
"""

import argparse
import random

import cv2

from configs.config import YOLO_DATASET_DIR


def draw_labels(image, label_path):
    img_h, img_w = image.shape[:2]
    n_boxes = 0
    for line in label_path.read_text(encoding="utf-8").splitlines():
        _, cx, cy, w, h = map(float, line.split())
        x1 = int((cx - w / 2) * img_w)  # inverse of to_yolo_line
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        n_boxes += 1
    return n_boxes


def main():
    parser = argparse.ArgumentParser(description="Display YOLO labels on random images")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=20, help="number of images to show")
    args = parser.parse_args()

    img_dir = YOLO_DATASET_DIR / "images" / args.split
    lbl_dir = YOLO_DATASET_DIR / "labels" / args.split
    images = sorted(img_dir.glob("*.jpg"))

    missing = [p.name for p in images if not (lbl_dir / f"{p.stem}.txt").is_file()]
    print(
        f"[INFO] {args.split}: {len(images)} images, {len(missing)} without label file"
    )

    for img_path in random.sample(images, min(args.n, len(images))):
        image = cv2.imread(str(img_path))
        n_boxes = draw_labels(image, lbl_dir / f"{img_path.stem}.txt")

        cv2.putText(
            image,
            f"{img_path.stem} | {n_boxes} boxes",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Check YOLO labels", image)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
