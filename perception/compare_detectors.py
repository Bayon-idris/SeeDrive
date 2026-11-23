"""
Compare the pre-trained YOLO (COCO) and the fine-tuned YOLO (UA-DETRAC) on the official UA-DETRAC test set, with the same metric (mAP).
"""

import torch
import torchvision
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

from ultralytics import YOLO

from configs.config import ROOT, YOLO_DATASET_DIR

COCO_VEHICLES = [2, 5, 7]
DATA = str(YOLO_DATASET_DIR / "dataset.yaml")
FINETUNED = ROOT / "runs" / "yolo11n_detrac" / "weights" / "best.pt"


#return only car / bus / truck -> vehicle 
def keep_vehicles(out, iou):

    keep = torch.isin(out["cls"], torch.tensor(COCO_VEHICLES, device=out["cls"].device))
    out = {k: v[keep] for k, v in out.items()}
    idx = torchvision.ops.nms(out["bboxes"], out["conf"], iou)
    out = {k: v[idx] for k, v in out.items()}
    out["cls"] = torch.zeros_like(out["cls"])
    return out


class COCOVehicleValidator(DetectionValidator):

    def postprocess(self, preds):
        return [keep_vehicles(out, self.args.iou) for out in super().postprocess(preds)]


def evaluate(model_path, validator=None):
    model = YOLO(str(model_path))
    metrics = model.val(
        data=DATA,
        split="test",
        imgsz=960,
        batch=6,
        workers=2,
        validator=validator,
        plots=False,
        verbose=False,
    )
    return metrics.box


if __name__ == "__main__":
    pretrained = evaluate("yolo11n.pt", validator=COCOVehicleValidator)
    finetuned = evaluate(FINETUNED)

    print(
        f"\n{'Detector':<22}{'Precision':>10}{'Recall':>8}{'mAP50':>8}{'mAP50-95':>10}"
    )
    for name, box in (
        ("YOLO11n pre-trained", pretrained),
        ("YOLO11n fine-tuned", finetuned),
    ):
        print(
            f"{name:<22}{100 * box.mp:>10.1f}{100 * box.mr:>8.1f}"
            f"{100 * box.map50:>8.1f}{100 * box.map:>10.1f}"
        )
