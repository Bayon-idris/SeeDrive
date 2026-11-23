from configs.config import (
    IMAGES_DIR,
    TEST_ANNOTATIONS_DIR,
    TRAIN_ANNOTATIONS_DIR,
    YOLO_DATASET_DIR,
)
import xml.etree.ElementTree as ET
from collections import defaultdict
import cv2

N_VAL = 6
FRAME_STEP = 10
CLASSES = ["car", "bus", "van", "others"]


def split_sequences():

    official_train = sorted(p.stem for p in TRAIN_ANNOTATIONS_DIR.glob("*.xml"))
    test = sorted(p.stem for p in TEST_ANNOTATIONS_DIR.glob("*.xml"))

    step = max(1, len(official_train) // N_VAL)
    val = official_train[::step][:N_VAL]  # spread picks across locations
    train = [s for s in official_train if s not in val]
    return train, val, test


def xml_path_for(seq):

    for folder in (TRAIN_ANNOTATIONS_DIR, TEST_ANNOTATIONS_DIR):
        path = folder / f"{seq}.xml"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No annotation file for {seq}")


def read_box(box):
    x, y = float(box.get("left")), float(box.get("top"))
    w, h = float(box.get("width")), float(box.get("height"))
    return (x, y, x + w, y + h)


def read_annotations(xml_path):

    root = ET.parse(xml_path).getroot()

    region = root.find("ignored_region")
    ignored = [read_box(b) for b in region.findall("box")] if region is not None else []

    boxes_per_frame = defaultdict(list)
    for frame in root.findall("frame"):
        target_list = frame.find("target_list")
        if target_list is None:  # frame with no vehicle
            continue
        frame_id = int(frame.get("num"))
        for target in target_list.findall("target"):
            cls = CLASSES.index(target.find("attribute").get("vehicle_type"))
            boxes_per_frame[frame_id].append((read_box(target.find("box")), cls))

    return boxes_per_frame, ignored


def to_yolo_line(box, img_w, img_h, cls=0):
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, x1), min(float(img_w), x2)  # clip boxes that go outside the image
    y1, y2 = max(0.0, y1), min(float(img_h), y2)
    if x2 <= x1 or y2 <= y1:  # box fully outside the image
        return None
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def convert_frame(seq, frame_id, boxes, ignored, split):
    src = IMAGES_DIR / seq / f"img{frame_id:05d}.jpg"
    image = cv2.imread(str(src))
    if image is None:
        print(f"[WARN] Cannot read {src}")
        return False
    img_h, img_w = image.shape[:2]

    # Paint gray zones black: their cars are not labeled, YOLO must not see them
    for x1, y1, x2, y2 in ignored:
        cv2.rectangle(
            image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), thickness=-1
        )

    name = f"{seq}_img{frame_id:05d}"  # unique name across all sequences
    img_dir = YOLO_DATASET_DIR / "images" / split
    lbl_dir = YOLO_DATASET_DIR / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(img_dir / f"{name}.jpg"), image)
    lines = [
        line
        for line in (to_yolo_line(b, img_w, img_h, cls) for b, cls in boxes)
        if line
    ]
    (lbl_dir / f"{name}.txt").write_text(
        "".join(l + "\n" for l in lines), encoding="utf-8"
    )
    return True


def build_split(sequences, split, frame_step=FRAME_STEP):
    n_images = n_boxes = 0
    for seq in sequences:
        boxes, ignored = read_annotations(xml_path_for(seq))
        n_frames = len(list((IMAGES_DIR / seq).glob("img*.jpg")))

        for frame_id in range(1, n_frames + 1, frame_step):  # 1, 11, 21, ...
            frame_boxes = boxes.get(frame_id, [])  # [] = no vehicle on this frame
            if convert_frame(seq, frame_id, frame_boxes, ignored, split):
                n_images += 1
                n_boxes += len(frame_boxes)

        print(f"  {split:<5} {seq} done")
    print(f"[OK] {split}: {n_images} images, {n_boxes} boxes\n")


def write_dataset_yaml():
    yaml_path = YOLO_DATASET_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"path: {YOLO_DATASET_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES)),
        encoding="utf-8",
    )
    print(f"[OK] {yaml_path}")


if __name__ == "__main__":
    train, val, test = split_sequences()
    build_split(train, "train")
    build_split(val, "val")
    build_split(test, "test")
    write_dataset_yaml()
