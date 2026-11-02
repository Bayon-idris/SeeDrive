"""
tracking/save_detections.py
Step 1: run YOLO (detection only, no tracking) on UA-DETRAC image sequences
and save all detections in MOT format, one text file per sequence.

Line format (MOT):  frame, -1, x, y, w, h, conf, -1, -1, -1
    frame : frame number (starts at 1, same as the image name img00001.jpg)
    -1    : no ID yet (the tracker will assign IDs later)
    x, y  : top-left corner of the box (pixels)
    w, h  : width and height of the box (pixels)
    conf  : detection confidence (0 to 1)

"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "DETRAC-Images" / "DETRAC-Images"
OUTPUT_DIR = ROOT / "outputs" / "detections"

DEFAULT_SEQUENCES = ["MVI_20011", "MVI_20012", "MVI_20032"]
VEHICLE_CLASSES = [2, 5, 7]          # COCO: car, bus, truck (UA-DETRAC only annotates vehicles)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def frame_number(image_path):
    """img00012.jpg -> 12"""
    digits = "".join(c for c in image_path.stem if c.isdigit())
    return int(digits)


def save_sequence(model, seq_name, imgsz, conf):
    seq_dir = DATASET_DIR / seq_name
    if not seq_dir.is_dir():
        print(f"[ERROR] Sequence not found: {seq_dir}")
        return

    images = sorted(
        (p for p in seq_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
        key=frame_number,
    )
    if not images:
        print(f"[ERROR] No images in {seq_dir}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{seq_name}.txt"
    n_boxes = 0

    print(f"[RUN] {seq_name}: {len(images)} frames...")

    with open(out_path, "w", encoding="utf-8") as f:
        for image_path in images:
            frame_id = frame_number(image_path)

            result = model.predict(
                str(image_path),
                classes=VEHICLE_CLASSES,
                imgsz=imgsz,
                conf=conf,
                verbose=False,
            )[0]

            boxes = result.boxes.xyxy.tolist()    # [x1, y1, x2, y2] per box
            scores = result.boxes.conf.tolist()

            for (x1, y1, x2, y2), score in zip(boxes, scores):
                w, h = x2 - x1, y2 - y1
                f.write(f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1\n")
                n_boxes += 1

    print(f"[OK] {seq_name}: {n_boxes} boxes saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Save YOLO detections in MOT format")
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.1,
                        help="low on purpose: ByteTrack uses low-confidence boxes, SORT will filter")
    args = parser.parse_args()

    model = YOLO(args.model)

    for seq_name in args.sequences:
        save_sequence(model, seq_name, args.imgsz, args.conf)

    print("\nDone.")


if __name__ == "__main__":
    main()