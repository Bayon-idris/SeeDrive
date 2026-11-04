
import argparse
from pathlib import Path

from configs.config import CONFIG_PATH
from ingest.rtsp_reader import load_cameras, start_cameras, stop_cameras
from perception.inference import  run_inference
from configs.helpers import load_model   


def main():
    
    
    model = load_model("yolo11n.pt")
    if model is None:
        return
    parser = argparse.ArgumentParser(
        description="Lecteur RTSP SeeDrive"
    )
    parser.add_argument("--camera", help="ID de la caméra")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--segment", type=float, default=60)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    cameras = load_cameras(Path(args.config))

    if not cameras:
        print("[ERREUR] Aucune caméra active.")
        return

    if args.camera:
        matches = [c for c in cameras if c["id"] == args.camera]
        if not matches:
            print(f"[ERREUR] Caméra inconnue : {args.camera}")
            return
        camera = matches[0]
    else:
        camera = cameras[0]

    processes = []

    try:
        processes = start_cameras(cameras)

        # run_camera(
        #     camera,
        #     segment=args.segment,
        #     max_duration=args.duration,
        #     record=not args.no_record,
        #     display=not args.no_display,
        # )
        
        run_inference(model, camera["rtsp_url"])

    except KeyboardInterrupt:
        print("\n[STOP] Arrêt demandé.")

    finally:
        stop_cameras(processes)


if __name__ == "__main__":
    main()
