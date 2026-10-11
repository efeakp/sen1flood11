"""PyTorch Dataset for SEN1Floods11 SAR chips."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


class SEN1FloodsDataset(Dataset):
    """
    Loads Sentinel-1 SAR chips (VV + VH) and binary flood masks.

    Args:
        split_csv: Path to the stratified split CSV from SEN1Floods11.
        data_dir:  Root directory containing S1Hand/ and LabelHand/ subdirs.
        split:     One of 'train', 'val', 'test'.
        transform: Albumentations transform applied to image + mask together.
    """

    def __init__(
        self,
        split_csv: str | Path,
        data_dir: str | Path,
        split: str = "train",
        transform=None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform

        df = pd.read_csv(split_csv)
        self.samples = df[df["split"] == split]["chip_id"].tolist()

        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found for split='{split}'. "
                f"Available splits: {df['split'].unique().tolist()}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        chip_id = self.samples[idx]

        image = self._load_sar(chip_id)    # (2, H, W) float32
        mask = self._load_mask(chip_id)    # (H, W) int64

        if self.transform:
            # albumentations expects HWC images
            augmented = self.transform(
                image=image.transpose(1, 2, 0),
                mask=mask,
            )
            image = augmented["image"].transpose(2, 0, 1)
            mask = augmented["mask"]

        return {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).long(),
            "chip_id": chip_id,
        }

    def _load_sar(self, chip_id: str) -> np.ndarray:
        path = self.data_dir / "S1Hand" / f"{chip_id}_S1Hand.tif"
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)  # (2, H, W)

        # Replace nodata with 0 and clip to reasonable SAR range
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data = np.clip(data, -50, 1)
        # Normalise to [0, 1]
        data = (data + 50) / 51
        return data

    def _load_mask(self, chip_id: str) -> np.ndarray:
        path = self.data_dir / "LabelHand" / f"{chip_id}_LabelHand.tif"
        with rasterio.open(path) as src:
            mask = src.read(1).astype(np.int64)  # (H, W)

        # -1 in SEN1Floods11 = no data; remap to 0 (not flooded)
        mask = np.where(mask == -1, 0, mask)
        return mask
