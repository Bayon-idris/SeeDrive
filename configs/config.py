"""
configs/config.py
Single place for every path and setting of SeeDrive.
Other files import from here:  from configs.config import ROOT, IMAGES_DIR, ...
"""

import sys
from pathlib import Path


# ============================================================
# 1. PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# 2. DATASET (UA-DETRAC)
# ============================================================

IMAGES_DIR = ROOT / "datasets" / "DETRAC-Images" / "DETRAC-Images"
TRAIN_ANNOTATIONS_DIR = (ROOT / "datasets" / "DETRAC-Train-Annotations-XML"
                         / "DETRAC-Train-Annotations-XML")
TEST_ANNOTATIONS_DIR = (ROOT / "datasets" / "DETRAC-Test-Annotations-XML"
                        / "DETRAC-Test-Annotations-XML")   

VIDEOS_DIR = ROOT / "datasets" / "videos" / "generated"      # .mp4 made by create_detrac_videos.py

DEFAULT_SEQUENCES = ["MVI_20011", "MVI_20012", "MVI_20032"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
FPS = 25                                                     # UA-DETRAC is filmed at 25 fps
YOLO_DATASET_DIR = ROOT / "datasets" / "detrac_yolo"         # fine-tuning dataset (YOLO format)


# ============================================================
# 3. LIVE PIPELINE (cameras, RTSP, logs)
# ============================================================

CONFIG_PATH = ROOT / "configs" / "cameras.yaml"
LOG_DIR = ROOT / "log"
DEFAULT_FPS = float(FPS)                                     # fallback when RTSP reports a wrong FPS


# ============================================================
# 4. DETECTION (YOLO, COCO class IDs)
# ============================================================

TARGET_CLASSES = [2, 3, 5, 7]     # live pipeline: car, motorcycle, bus, truck
VEHICLE_CLASSES = [2, 5, 7]       # evaluation: car, bus, truck (what UA-DETRAC annotates)


# ============================================================
# 5. TRACKING BENCH (outputs)
# ============================================================

OUTPUTS_DIR = ROOT / "outputs"
DETECTIONS_DIR = OUTPUTS_DIR / "detections"                  # YOLO boxes without IDs
IGNORED_DIR = OUTPUTS_DIR / "ignored_regions"                # gray zones not annotated by humans

# TrackEval folder layout: <gt or trackers>/mot_challenge/DETRAC-train/...
BENCHMARK = "DETRAC"
SPLIT = "train"
BENCHMARK_SPLIT = f"{BENCHMARK}-{SPLIT}"                     # "DETRAC-train"

TRACKEVAL_DIR = OUTPUTS_DIR / "trackeval"
GT_FOLDER = TRACKEVAL_DIR / "gt" / "mot_challenge"           # ground truth root (for TrackEval)
TRACKERS_FOLDER = TRACKEVAL_DIR / "trackers" / "mot_challenge"
GT_DIR = GT_FOLDER / BENCHMARK_SPLIT                         # one folder per sequence: <SEQ>/gt/gt.txt
TRACKERS_DIR = TRACKERS_FOLDER / BENCHMARK_SPLIT             # one folder per tracker: <NAME>/data/<SEQ>.txt


# ============================================================
# 6. C++ SORT TRACKER
# ============================================================

# Folder that contains sort_tracker.pyd (built with CMake)
SORT_BUILD_DIR = ROOT / "tracking" / "sort_cpp" / "build" / "Release"

# Lets "import sort_tracker" work in any file that imports this config first
if str(SORT_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(SORT_BUILD_DIR))


# ============================================================
# 7. PERFORMANCE MONITORING
# ============================================================

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


# ============================================================
# 8. OLD NAMES (kept so existing scripts don't break)
# ============================================================
# Replace them little by little with the new names, then delete this section.

DATASET_DIR = IMAGES_DIR             # new name: IMAGES_DIR
OUTPUT_DIR = DETECTIONS_DIR          # new name: DETECTIONS_DIR
TRACKEVAL_GT_DIR = GT_FOLDER         # new name: GT_FOLDER
ANNOTATIONS_DIR = TRAIN_ANNOTATIONS_DIR   # new name: TRAIN_ANNOTATIONS_DIR