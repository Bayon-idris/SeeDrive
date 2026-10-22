from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent         
CONFIG_PATH = ROOT / "configs" / "cameras.yaml"
LOG_DIR = ROOT / "log"
DEFAULT_FPS = 25.0   



TARGET_CLASSES = [0, 2, 3, 5, 7]

# Display order of the latency table
STAGES = [
    "read",
    "inference",
    "yolo_preprocess",
    "yolo_inference",
    "yolo_postprocess",
    "display",
    "end_to_end",
]
