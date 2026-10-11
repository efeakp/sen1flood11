"""
Download SEN1Floods11 dataset via Radiant MLHub API.

Requires a free API key from https://mlhub.earth

Usage:
    export MLHUB_API_KEY=your_key_here
    python scripts/download_data.py --split hand_labeled --out data/raw
"""

import argparse
import os
import sys
from pathlib import Path

MLHUB_COLLECTION = "sen1floods11_v1.1"

HAND_LABELED_ASSETS = ["s1hand", "labelhand"]
WEAKLY_LABELED_ASSETS = ["s1weak", "s2weak", "labelweak"]


def download_mlhub(split: str, out_dir: Path) -> None:
    api_key = os.environ.get("MLHUB_API_KEY")
    if not api_key:
        sys.exit(
            "Error: set MLHUB_API_KEY environment variable.\n"
            "  export MLHUB_API_KEY=your_key_here\n"
            "Register for a free key at https://mlhub.earth"
        )

    try:
        from radiant_mlhub import Dataset
    except ImportError:
        sys.exit("Install radiant-mlhub: pip install radiant-mlhub")

    out_dir.mkdir(parents=True, exist_ok=True)

    asset_filter = HAND_LABELED_ASSETS if split == "hand_labeled" else WEAKLY_LABELED_ASSETS

    print(f"Fetching collection '{MLHUB_COLLECTION}' from Radiant MLHub...")
    dataset = Dataset.fetch(MLHUB_COLLECTION, api_key=api_key)

    matched = [
        c for c in dataset.collections
        if any(a in c.id.lower() for a in asset_filter)
    ]

    if not matched:
        print(f"Available collections: {[c.id for c in dataset.collections]}")
        sys.exit("No matching collections found. Check asset names above.")

    for collection in matched:
        print(f"Downloading collection: {collection.id}")
        collection.download(output_dir=str(out_dir))

    print(f"\nDone. Data saved to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SEN1Floods11 via Radiant MLHub")
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

    print(f"Split: {args.split} | Output: {args.out}\n")
    download_mlhub(args.split, args.out)


if __name__ == "__main__":
    main()
