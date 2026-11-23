import re
import subprocess

from configs.config import DATASET_DIR, FPS, IMAGE_EXTENSIONS, VIDEOS_DIR


def natural_sort_key(path):
    
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def create_video(sequence_dir):

    images = [
        p
        for p in sequence_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        print(f"[IGNORÉ] Aucune image dans {sequence_dir.name}")
        return

    output_path = VIDEOS_DIR / f"{sequence_dir.name}.mp4"

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(FPS),
        "-i",
        str(sequence_dir / "img%05d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(FPS),
        str(output_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERREUR] {sequence_dir.name} : {result.stderr.strip()}")
        return

    duration = len(images) / FPS
    print(
        f"[OK] {sequence_dir.name} -> {output_path.name} | "
        f"{len(images)} images | {duration:.2f} secondes"
    )


def main():
    if not DATASET_DIR.is_dir():
        print(f"[ERREUR] Dossier introuvable : {DATASET_DIR}")
        return

    sequences = sorted(
        [p for p in DATASET_DIR.iterdir() if p.is_dir()], key=natural_sort_key
    )

    print(f"Nombre de séquences trouvées : {len(sequences)}")
    print(f"Dossier de sortie : {VIDEOS_DIR}")
    print(f"FPS des vidéos : {FPS}\n")

    for sequence_dir in sequences:
        print(f"Traitement de {sequence_dir.name}...")
        create_video(sequence_dir)

    print("\nConversion terminée.")


if __name__ == "__main__":
    main()
