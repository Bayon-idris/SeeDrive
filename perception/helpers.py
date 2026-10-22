from ultralytics import YOLO


def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None
