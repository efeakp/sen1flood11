"""
Download SEN1Floods11 dataset.

Supports two methods:
  --method gsutil   Google Cloud Storage (recommended, no account needed)
  --method hf       Hugging Face datasets (alternative, no account needed)

Usage:
    python scripts/download_data.py --method gsutil --out data/raw
    python scripts/download_data.py --method hf     --out data/raw
"""

import argparse
import subprocess
import sys
from pathlib import Path

GCS_BUCKET = "gs://sen1floods11"


def download_gsutil(out_dir: Path) -> None:
    _check_gsutil()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SEN1Floods11 from {GCS_BUCKET} -> {out_dir}")
    print("This is ~14 GB for the full dataset. Ctrl+C to cancel.\n")
    _run(["gsutil", "-m", "rsync", "-r", GCS_BUCKET, str(out_dir)])
    print(f"\nDone. Data saved to {out_dir}/")


def download_hf(out_dir: Path) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install huggingface datasets: pip install datasets")

    out_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading SEN1Floods11 from Hugging Face (harshinde/sen1floods11)...")
    ds = load_dataset("harshinde/sen1floods11", cache_dir=str(out_dir))
    print(f"\nDone. Dataset cached to {out_dir}/")
    print(f"Splits available: {list(ds.keys())}")


def _check_gsutil() -> None:
    if subprocess.run(["which", "gsutil"], capture_output=True).returncode != 0:
        sys.exit(
            "gsutil not found. Install it:\n"
            "  pip install gsutil\n"
            "Or use --method hf instead."
        )


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"Command failed: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SEN1Floods11 dataset")
    parser.add_argument(
        "--method",
        choices=["gsutil", "hf"],
        default="gsutil",
        help="Download method: gsutil (GCS) or hf (Hugging Face). Default: gsutil",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw"),
        help="Output directory (default: data/raw)",
    )
    args = parser.parse_args()

    print(f"Method: {args.method} | Output: {args.out}\n")

    if args.method == "gsutil":
        download_gsutil(args.out)
    else:
        download_hf(args.out)


if __name__ == "__main__":
    main()
