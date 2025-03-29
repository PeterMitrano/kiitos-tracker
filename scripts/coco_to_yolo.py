import argparse
from pathlib import Path

from ultralytics.data.converter import convert_coco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels_dir", type=Path, default=Path("dataset"),
        help="directory containing the .json file"
    )

    args = parser.parse_args()

    assert args.labels_dir.exists()

    convert_coco(labels_dir=args.labels_dir, cls91to80=False, use_segments=True)

if __name__ == "__main__":
    main()
