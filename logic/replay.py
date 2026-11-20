"""
logic/replay.py
Replay a recorded sequence through the full traffic logic (speeds, counts, rules, alerts).
Keys: SPACE = pause / resume, q = quit

Run from the project root:
    python -m logic.replay --camera cam_01
    python -m logic.replay --camera cam_01 --simulate          # demo: wrong-way alerts
    python -m logic.replay --camera cam_01 --simulate --save   # also write a demo video
"""

import argparse
from collections import defaultdict

import cv2

from configs.config import FPS, IMAGES_DIR, OUTPUTS_DIR, TRACKERS_DIR
from logic.engine import TrafficLogic
from logic.speed import load_tracks


def main():
    parser = argparse.ArgumentParser(description="Replay a sequence with the traffic logic")
    parser.add_argument("--camera", default="cam_01")
    parser.add_argument("--simulate", action="store_true", help="swap expected directions (demo)")
    parser.add_argument("--save", action="store_true", help="write outputs/demo_<camera>.mp4")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    logic = TrafficLogic(args.camera, simulate=args.simulate)
    seq = logic.scene["sequence"]

    per_frame = defaultdict(list)                 # what the tracker gave at each frame
    for track_id, rows in load_tracks(TRACKERS_DIR / "SORT" / "data" / f"{seq}.txt").items():
        for frame, *box in rows:
            per_frame[frame].append((track_id, box))

    writer = None
    paused = False
    for image_path in sorted((IMAGES_DIR / seq).glob("img*.jpg")):
        frame = int(image_path.stem[3:])          # "img00012" -> 12
        image = cv2.imread(str(image_path))

        logic.update(frame, per_frame.get(frame, []))
        logic.draw(image, frame)

        if args.save:
            if writer is None:
                OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
                out_path = OUTPUTS_DIR / f"demo_{args.camera}.mp4"
                h, w = image.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
            writer.write(image)

        if not args.no_display:
            cv2.imshow("SeeDrive - traffic logic", image)
            key = cv2.waitKey(0 if paused else 1000 // FPS) & 0xFF
            if key == ord(" "):
                paused = not paused
            elif key == ord("q"):
                break

    if writer is not None:
        writer.release()
        print(f"[OK] Demo video saved -> {out_path}")
    cv2.destroyAllWindows()
    print("Counts:", logic.counter.counts)


if __name__ == "__main__":
    main()