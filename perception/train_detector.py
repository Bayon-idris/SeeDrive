from ultralytics import YOLO

from configs.config import ROOT, YOLO_DATASET_DIR


def main():
    model = YOLO("yolo11n.pt")  # start from the COCO weights (= fine-tuning)

    model.train(
        data=str(YOLO_DATASET_DIR / "dataset.yaml"),
        imgsz=960,  # same size as the live pipeline: far vehicles stay visible
        epochs=50,
        patience=10,  # stop early if val mAP does not improve for 10 epochs
        batch=6,
        device=0,
        workers=2,
        fraction=1.0,
        project=str(ROOT / "runs"),
        name="yolo11n_detrac",
        exist_ok=True,  # overwrite previous runs with the same name
    )


if __name__ == "__main__":
    main()
