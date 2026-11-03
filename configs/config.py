from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent         
CONFIG_PATH = ROOT / "configs" / "cameras.yaml"
LOG_DIR = ROOT / "log"
DEFAULT_FPS = 25.0   


DETECTIONS_DIR = ROOT / "outputs" / "detections"
# RESULTS_DIR = (ROOT / "outputs" / "trackeval" / "trackers" / "mot_challenge"
#                / "DETRAC-train" / "SORT" / "data")

TRACKERS_DIR = (ROOT / "outputs" / "trackeval" / "trackers" / "mot_challenge"
                / "DETRAC-train")

# Folder that contains sort_tracker.pyd (built with CMake)
SORT_BUILD_DIR = ROOT / "tracking" / "sort_cpp" / "build" / "Release"
sys.path.insert(0, str(SORT_BUILD_DIR))
IGNORED_DIR = ROOT / "outputs" / "ignored_regions"


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


# Dossier contenant toutes les séquences UA-DETRAC
DATASET_DIR = Path(
    r"D:\Projects\SeeDrive\datasets\DETRAC-Images\DETRAC-Images"
)

# Dossier de sortie des vidéos
OUTPUT_DIR = Path(r"D:\Projects\SeeDrive\datasets\videos\generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Vitesse de lecture des vidéos générées
FPS = 25

# Extensions d'images acceptées
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}