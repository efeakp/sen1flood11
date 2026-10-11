"""
Download SEN1Floods11 dataset.

Supports two methods:
  --method gsutil   Direct download from Google Cloud Storage (no account needed)
  --method mlhub    Radiant MLHub API (requires MLHUB_API_KEY env var)

Usage:
    python scripts/download_data.py --method gsutil --split hand_labeled --out data/raw
    python scripts/download_data.py --method mlhub  --split hand_labeled --out data/raw
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

GCS_BUCKET = "gs://sen1-floods11-a8d7/v1.1"
MLHUB_COLLECTION = "sen1floods11_v1.1"

HAND_LABELED_ASSETS = ["S1Hand", "LabelHand"]
WEAKLY_LABELED_ASSETS = ["S1Weak", "S2Weak", "LabelWeak"]

SPLIT_CSV_PATH = (
    f"{GCS_BUCKET}/splits/flood_handlabeled_split_stratified.csv"
)


def download_gsutil(split: str, out_dir: Path) -> None:
    _check_gsutil()
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = HAND_LABELED_ASSETS if split == "hand_labeled" else WEAKLY_LABELED_ASSETS

    # Download split CSV first
    print("Downloading split CSV...")
    _run(["gsutil", "-m", "cp", SPLIT_CSV_PATH, str(out_dir)])

    for asset in assets:
        src = f"{GCS_BUCKET}/data/flood_events/{asset}/"
        dst = str(out_dir / asset)
        print(f"\nDownloading {asset} -> {dst}")
        _run(["gsutil", "-m", "cp", "-r", src, dst])

    print(f"\nDone. Data saved to {out_dir}/")


def download_mlhub(split: str, out_dir: Path) -> None:
    api_key = os.environ.get("MLHUB_API_KEY")
    if not api_key:
        sys.exit(
            "Error: set MLHUB_API_KEY environment variable.\n"
            "Register at https://mlhub.earth to get a key."
        )

    try:
        from radiant_mlhub import Dataset
    except ImportError:
        sys.exit("Install radiant-mlhub: pip install radiant-mlhub")

    out_dir.mkdir(parents=True, exist_ok=True)

    assets = HAND_LABELED_ASSETS if split == "hand_labeled" else WEAKLY_LABELED_ASSETS

    print(f"Fetching collection {MLHUB_COLLECTION} from Radiant MLHub...")
    dataset = Dataset.fetch(MLHUB_COLLECTION, api_key=api_key)

    for collection in dataset.collections:
        if any(a.lower() in collection.id.lower() for a in [a.lower() for a in assets]):
            print(f"Downloading collection: {collection.id}")
            collection.download(output_dir=str(out_dir))

    print(f"\nDone. Data saved to {out_dir}/")


def _check_gsutil() -> None:
    if subprocess.run(["which", "gsutil"], capture_output=True).returncode != 0:
        sys.exit(
            "gsutil not found. Install it:\n"
            "  https://cloud.google.com/storage/docs/gsutil_install\n"
            "Or use --method mlhub instead."
        )


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"Command failed: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SEN1Floods11 dataset")
    parser.add_argument(
        "--method",
        choices=["gsutil", "mlhub"],
        default="gsutil",
        help="Download method (default: gsutil)",
    )
    parser.add_argument(
        "--split",
        choices=["hand_labeled", "weakly_labeled"],
        default="hand_labeled",
        help="Which labeled subset to download (default: hand_labeled)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw"),
        help="Output directory (default: data/raw)",
    )
    args = parser.parse_args()

    print(f"Method: {args.method} | Split: {args.split} | Output: {args.out}\n")

    if args.method == "gsutil":
        download_gsutil(args.split, args.out)
    else:
        download_mlhub(args.split, args.out)


if __name__ == "__main__":
    main()
