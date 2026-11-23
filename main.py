import argparse

from perception.live import run


def main():
    parser = argparse.ArgumentParser(
        description="SeeDrive - real-time traffic analysis"
    )
    parser.add_argument(
        "--camera", default="cam_01", help="camera id from configs/cameras.yaml"
    )
    parser.add_argument(
        "--simulate", action="store_true", help="demo mode: wrong-way alerts"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="record the display in outputs/demo_<camera>.mp4",
    )
    args = parser.parse_args()
    run(args.camera, simulate=args.simulate, save=args.save)


if __name__ == "__main__":
    main()
