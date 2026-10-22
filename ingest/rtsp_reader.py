import os
import time
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import shutil

from configs.config import DEFAULT_FPS, LOG_DIR, ROOT
from test import CONFIG_PATH

# RTSP en TCP : évite les images grises/corrompues dues aux paquets UDP perdus.
# Doit être défini AVANT l'import de cv2.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import yaml

                                  
def load_cameras(config_path=CONFIG_PATH, only_enabled=True):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    cameras = []
    for i, cam in enumerate(config.get("cameras", [])):
        if "rtsp_url" not in cam:
            print(f"[CONFIG] Caméra #{i} ignorée : pas de rtsp_url")
            continue

        camera = {
            "id": cam.get("id", f"cam_{i + 1:02d}"),
            "rtsp_url": cam["rtsp_url"],
            "source_video": cam.get("source_video"),
            "enabled": cam.get("enabled", True),
        }
        if only_enabled and not camera["enabled"]:
            continue
        cameras.append(camera)

    return cameras


def start_cameras(cameras):
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        raise RuntimeError("FFmpeg est introuvable dans le PATH.")

    processes = []

    for camera in cameras:
        camera_id = camera["id"]
        source = camera.get("source_video")
        rtsp_url = camera["rtsp_url"]

        if not source:
            print(f"[ERREUR] {camera_id}: source_video manquant.")
            continue

        source_video = ROOT / source

        if not source_video.is_file():
            print(f"[ERREUR] Vidéo introuvable : {source_video}")
            continue

        command = [
            ffmpeg,
            "-re",
            "-stream_loop", "-1",
            "-i", str(source_video),
            "-c", "copy",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            rtsp_url,
        ]

        print(f"[FFMPEG] Démarrage de {camera_id}: {rtsp_url}")

        try:
            process = subprocess.Popen(command)
            processes.append((camera_id, process))
        except OSError as exc:
            print(f"[ERREUR] {camera_id}: {exc}")

    return processes


def stop_cameras(processes):
    for camera_id, process in processes:
        if process.poll() is None:
            print(f"[STOP] {camera_id}")
            process.terminate()

    for camera_id, process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def open_stream(url, retries=5, delay=2.0):
    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            return cap
        cap.release()
        print(f"[RTSP] Échec ouverture {url} ({attempt}/{retries}), nouvel essai dans {delay}s")
        time.sleep(delay)
    return None


def safe_fps(cap):
    
    """En RTSP, CAP_PROP_FPS renvoie souvent 0 ou une valeur absurde (ex. 90000)."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 120:
        return DEFAULT_FPS
    return fps


def new_writer(cam_id, fps, size, output_dir=LOG_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = output_dir / f"{cam_id}_{dt}.mp4"           
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    print(f"[REC] Nouveau segment : {path}")
    return writer


def run_camera(camera, segment=60, max_duration=None, record=True, display=True,
               max_failures=25):
    """
    Lit une caméra en continu.
    - segment      : durée (s) de chaque fichier enregistré
    - max_duration : arrêt après N secondes (None = jusqu'à 'q' ou Ctrl+C)
    """
    cam_id, url = camera["id"], camera["rtsp_url"]

    cap = open_stream(url)
    if cap is None:
        print(f"[ERREUR] Impossible d'ouvrir {cam_id} ({url})")
        return

    fps = safe_fps(cap)
    writer = None
    seg_start = 0.0
    size = None

    start = time.time()
    frame_count = 0
    failures = 0

    # FPS glissant (recalculé chaque seconde) au lieu d'une moyenne depuis le début
    fps_t0, fps_frames, fps_display = time.time(), 0, 0.0

    try:
        while True:
            now = time.time()
            if max_duration is not None and now - start >= max_duration:
                break

            ret, frame = cap.read()

            if not ret:
                failures += 1
                if failures >= max_failures:
                    print(f"[RTSP] {cam_id} : flux perdu, reconnexion...")
                    cap.release()
                    cap = open_stream(url)
                    if cap is None:
                        print(f"[ERREUR] {cam_id} : reconnexion impossible, arrêt.")
                        break
                    failures = 0
                continue
            failures = 0
            frame_count += 1

            # taille réelle lue sur l'image (CAP_PROP_FRAME_WIDTH peut valoir 0 en RTSP)
            if size is None:
                size = (frame.shape[1], frame.shape[0])

            # ---- enregistrement, avec rotation de fichier toutes les `segment` secondes
            if record:
                if writer is None or now - seg_start >= segment:
                    if writer is not None:
                        writer.release()
                    writer = new_writer(cam_id, fps, size)
                    seg_start = now
                writer.write(frame)                    # image brute, avant l'incrustation du FPS

            fps_frames += 1
            if now - fps_t0 >= 1.0:
                fps_display = fps_frames / (now - fps_t0)
                fps_t0, fps_frames = now, 0

            if display:
                cv2.putText(frame, f"{cam_id} | FPS: {fps_display:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(cam_id, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[STOP] Interruption clavier.")

    finally:
        if writer is not None:
            writer.release()
        cap.release()
        if display:
            cv2.destroyAllWindows()
        print(f"[OK] {cam_id} : {frame_count} frames lues en {time.time() - start:.1f}s")

