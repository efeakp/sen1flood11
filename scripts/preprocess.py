"""
Validate downloaded chips and print dataset statistics.
Run after download_data.py to confirm data integrity.

Usage:
    python scripts/preprocess.py --data data/raw
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm


def validate_and_stats(data_dir: Path) -> None:
    s1_dir = data_dir / "S1Hand"
    label_dir = data_dir / "LabelHand"

    s1_chips = sorted(s1_dir.glob("*_S1Hand.tif"))
    label_chips = sorted(label_dir.glob("*_LabelHand.tif"))

    print(f"S1 chips:    {len(s1_chips)}")
    print(f"Label chips: {len(label_chips)}")

    s1_ids = {f.name.replace("_S1Hand.tif", "") for f in s1_chips}
    label_ids = {f.name.replace("_LabelHand.tif", "") for f in label_chips}
    missing_labels = s1_ids - label_ids
    missing_s1 = label_ids - s1_ids

    if missing_labels:
        print(f"WARNING: {len(missing_labels)} S1 chips have no label.")
    if missing_s1:
        print(f"WARNING: {len(missing_s1)} label chips have no S1 image.")

    vv_means, vh_means, flood_fracs = [], [], []

    for s1_path in tqdm(s1_chips[:200], desc="Computing stats (first 200 chips)"):
        chip_id = s1_path.name.replace("_S1Hand.tif", "")
        label_path = label_dir / f"{chip_id}_LabelHand.tif"

        with rasterio.open(s1_path) as src:
            data = src.read().astype(np.float32)
            data = np.nan_to_num(data)

        vv_means.append(float(np.mean(data[0])))
        vh_means.append(float(np.mean(data[1])))

        if label_path.exists():
            with rasterio.open(label_path) as lsrc:
                mask = lsrc.read(1).astype(np.float32)
            valid = mask[mask >= 0]
            if len(valid) > 0:
                flood_fracs.append(float(np.mean(valid > 0)))

    print(f"\n--- Dataset Statistics (first 200 chips) ---")
    print(f"VV backscatter mean:  {np.mean(vv_means):.2f} dB")
    print(f"VH backscatter mean:  {np.mean(vh_means):.2f} dB")
    if flood_fracs:
        print(f"Flood pixel fraction: {np.mean(flood_fracs):.3f}")
        print(f"  (class imbalance — flooded pixels are ~{np.mean(flood_fracs)*100:.1f}% of all pixels)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    validate_and_stats(args.data)


if __name__ == "__main__":
    main()
