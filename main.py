"""
main.py
SeeDrive entry point: full live pipeline on one camera
(RTSP -> fine-tuned YOLO -> C++ SORT -> speed -> rules -> alerts).

    python main.py --camera cam_01
    python main.py --camera cam_01 --simulate     # demo: wrong-way alerts
"""

import argparse

from perception.live import run


def main():
    parser = argparse.ArgumentParser(description="SeeDrive - real-time traffic analysis")
    parser.add_argument("--camera", default="cam_01", help="camera id from configs/cameras.yaml")
    parser.add_argument("--simulate", action="store_true", help="demo mode: wrong-way alerts")
    args = parser.parse_args()
    run(args.camera, simulate=args.simulate)


if __name__ == "__main__":
    main()