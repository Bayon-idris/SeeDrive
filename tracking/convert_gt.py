"""
tracking/convert_gt.py
Step 2: convert UA-DETRAC XML annotations (ground truth) to MOT format,
directly in the folder structure expected by TrackEval.

Output:
    outputs/trackeval/gt/mot_challenge/DETRAC-train/<SEQ>/gt/gt.txt
    outputs/trackeval/gt/mot_challenge/DETRAC-train/<SEQ>/seqinfo.ini
    outputs/trackeval/gt/mot_challenge/seqmaps/DETRAC-train.txt
    outputs/ignored_regions/<SEQ>.txt

gt.txt line format:  frame, id, x, y, w, h, 1, 1, 1
    the last three values = "evaluate this box", class 1 (single class), fully visible

"""

import argparse
import xml.etree.ElementTree as ET

import cv2

from configs.config import (
    ANNOTATIONS_DIR,
    BENCHMARK_SPLIT,
    DEFAULT_SEQUENCES,
    FPS,
    IMAGE_EXTENSIONS,
    IMAGES_DIR,
    TRACKEVAL_GT_DIR,
)
from tracking.check_gt import IGNORED_DIR


def read_box(box):
    return tuple(float(box.get(k)) for k in ("left", "top", "width", "height"))


def sequence_info(seq_dir):
    """Number of frames and image size of a sequence."""
    images = sorted(
        p for p in seq_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    first = cv2.imread(str(images[0]))
    height, width = first.shape[:2]
    return len(images), width, height


def convert_sequence(seq_name):
    xml_path = ANNOTATIONS_DIR / f"{seq_name}.xml"
    seq_dir = IMAGES_DIR / seq_name

    if not xml_path.is_file():
        print(f"[ERROR] Annotation not found: {xml_path}")
        return False
    if not seq_dir.is_dir():
        print(f"[ERROR] Images not found: {seq_dir}")
        return False

    root = ET.parse(xml_path).getroot()

    # ---- ignored regions (areas not annotated: parked cars, far away traffic...)
    ignored = []
    region = root.find("ignored_region")
    if region is not None:
        ignored = [read_box(box) for box in region.findall("box")]

    # ---- annotated vehicles, frame by frame
    rows = []
    for frame in root.findall("frame"):
        frame_id = int(frame.get("num"))
        target_list = frame.find("target_list")
        if target_list is None:
            continue
        for target in target_list.findall("target"):
            track_id = int(target.get("id"))
            x, y, w, h = read_box(target.find("box"))
            rows.append((frame_id, track_id, x, y, w, h))
    rows.sort()

    # ---- gt.txt
    seq_out = TRACKEVAL_GT_DIR / BENCHMARK_SPLIT / seq_name
    (seq_out / "gt").mkdir(parents=True, exist_ok=True)
    with open(seq_out / "gt" / "gt.txt", "w", encoding="utf-8") as f:
        for frame_id, track_id, x, y, w, h in rows:
            f.write(f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n")

    # ---- seqinfo.ini (sequence description required by TrackEval)
    n_frames, width, height = sequence_info(seq_dir)
    with open(seq_out / "seqinfo.ini", "w", encoding="utf-8") as f:
        f.write(
            "[Sequence]\n"
            f"name={seq_name}\n"
            "imDir=img1\n"
            f"frameRate={FPS}\n"
            f"seqLength={n_frames}\n"
            f"imWidth={width}\n"
            f"imHeight={height}\n"
            "imExt=.jpg\n"
        )

    # ---- ignored regions (used later to clean tracker outputs)
    IGNORED_DIR.mkdir(parents=True, exist_ok=True)
    with open(IGNORED_DIR / f"{seq_name}.txt", "w", encoding="utf-8") as f:
        for x, y, w, h in ignored:
            f.write(f"{x:.2f},{y:.2f},{w:.2f},{h:.2f}\n")

    n_ids = len({r[1] for r in rows})
    print(
        f"[OK] {seq_name}: {n_frames} frames | {len(rows)} boxes | "
        f"{n_ids} vehicles | {len(ignored)} ignored regions"
    )
    return True


def write_seqmap(sequences):
    """List of sequences to evaluate (required by TrackEval)."""
    seqmap_dir = TRACKEVAL_GT_DIR / "seqmaps"
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    with open(seqmap_dir / f"{BENCHMARK_SPLIT}.txt", "w", encoding="utf-8") as f:
        f.write("name\n")
        for seq in sequences:
            f.write(f"{seq}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert UA-DETRAC XML to MOT format")
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    args = parser.parse_args()

    done = [seq for seq in args.sequences if convert_sequence(seq)]
    if done:
        write_seqmap(done)
        print(f"\nSeqmap written with {len(done)} sequence(s).")


if __name__ == "__main__":
    main()
