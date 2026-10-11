"""Generate flood frequency maps by aggregating per-event predictions."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds


def aggregate_predictions(
    prediction_dir: str | Path,
    output_path: str | Path,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Aggregate binary flood predictions across all event chips.

    For each spatial chip, counts how many times it was classified as flooded
    and divides by total events to get a flood frequency in [0, 1].

    Args:
        prediction_dir: Directory of .npy files, one per chip, containing
                        sigmoid probability arrays of shape (H, W).
        output_path:    Where to save the aggregated frequency map (.npy).
        threshold:      Probability cutoff for flood/no-flood classification.

    Returns:
        freq_map: (num_chips,) array of flood frequencies per chip.
    """
    prediction_dir = Path(prediction_dir)
    pred_files = sorted(prediction_dir.glob("*.npy"))

    if not pred_files:
        raise FileNotFoundError(f"No .npy prediction files found in {prediction_dir}")

    flood_counts = None
    total = len(pred_files)

    for f in pred_files:
        prob_map = np.load(f)             # (H, W) float32
        binary = (prob_map > threshold).astype(np.uint8)

        if flood_counts is None:
            flood_counts = np.zeros_like(binary, dtype=np.float32)
        flood_counts += binary

    freq_map = flood_counts / total
    np.save(output_path, freq_map)
    print(f"Saved frequency map to {output_path} (shape={freq_map.shape})")
    return freq_map


def frequency_map_to_geotiff(
    freq_map: np.ndarray,
    reference_tif: str | Path,
    output_path: str | Path,
) -> None:
    """Write a frequency map array as a GeoTIFF, georeferenced from a reference chip."""
    with rasterio.open(reference_tif) as ref:
        profile = ref.profile.copy()

    profile.update(dtype=rasterio.float32, count=1, compress="lzw")

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(freq_map.astype(np.float32), 1)

    print(f"Saved GeoTIFF to {output_path}")
