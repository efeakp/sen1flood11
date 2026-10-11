"""PyTorch Dataset for SEN1Floods11 SAR chips."""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


class SEN1FloodsDataset(Dataset):
    """
    Loads Sentinel-1 SAR chips (VV + VH) and binary flood masks.

    Args:
        split_csv: Path to one of the split CSVs (flood_train_data.csv, etc.).
                   Each row: S1Hand_filename,LabelHand_filename (no header).
        data_dir:  Root directory containing S1Hand/ and LabelHand/ subdirs.
        transform: Albumentations transform applied to image + mask together.
    """

    def __init__(
        self,
        split_csv: str | Path,
        data_dir: str | Path,
        transform=None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform

        df = pd.read_csv(split_csv, header=None, names=["s1", "label"])
        self.samples = list(zip(df["s1"], df["label"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s1_name, label_name = self.samples[idx]

        image = self._load_sar(s1_name)     # (2, H, W) float32
        mask = self._load_mask(label_name)  # (H, W) int64

        if self.transform:
            augmented = self.transform(
                image=image.transpose(1, 2, 0),
                mask=mask,
            )
            image = augmented["image"].transpose(2, 0, 1)
            mask = augmented["mask"]

        chip_id = s1_name.replace("_S1Hand.tif", "")
        return {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).long(),
            "chip_id": chip_id,
        }

    def _load_sar(self, filename: str) -> np.ndarray:
        path = self.data_dir / "S1Hand" / filename
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)  # (2, H, W)

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data = np.clip(data, -50, 1)
        data = (data + 50) / 51  # normalise to [0, 1]
        return data

    def _load_mask(self, filename: str) -> np.ndarray:
        path = self.data_dir / "LabelHand" / filename
        with rasterio.open(path) as src:
            mask = src.read(1).astype(np.int64)  # (H, W)

        # -1 = no data in SEN1Floods11; treat as not flooded
        mask = np.where(mask == -1, 0, mask)
        return mask
