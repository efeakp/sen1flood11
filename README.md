# Flood Monitoring with SEN1Floods11

Machine learning methods to monitor the **extent and frequency of flooding** using Sentinel-1 SAR imagery from the [SEN1Floods11](https://github.com/cloudtostreet/SEN1Floods11) benchmark dataset.

## Overview

- **Flood extent mapping** — semantic segmentation (U-Net) to classify flooded pixels per SAR image
- **Flood frequency analysis** — aggregate per-event predictions to produce spatial flood frequency/risk maps
- **Baselines** — VV backscatter threshold and Random Forest for comparison

## Dataset

SEN1Floods11 contains 4,831 hand-labeled 512×512 Sentinel-1 SAR chips across 14 flood events on 6 continents.

- 2-channel input: VV and VH polarization backscatter
- Binary labels: flooded / not flooded

## Project Structure

```
sen1flood11/
├── data/
│   ├── raw/            # downloaded GeoTIFFs (not tracked by git)
│   └── processed/      # preprocessed tensors (not tracked by git)
├── src/
│   ├── dataset.py      # PyTorch Dataset for SAR chips
│   ├── model.py        # U-Net model definition
│   ├── train.py        # training loop
│   ├── evaluate.py     # metrics (IoU, F1)
│   └── frequency.py    # flood frequency map generation
├── notebooks/
│   └── 01_eda.ipynb    # exploratory data analysis
├── scripts/
│   ├── download_data.py  # fetch SEN1Floods11 from GCS / MLHub
│   └── preprocess.py     # chip validation and stats
├── configs/
│   └── train_config.yaml
└── requirements.txt
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

**Option A — Google Cloud Storage (recommended, no account needed):**
```bash
python scripts/download_data.py --method gsutil
```

**Option B — Hugging Face (alternative, no account needed):**
```bash
pip install datasets
python scripts/download_data.py --method hf
```

> Full dataset is ~14 GB. Both methods download everything to `data/raw/`.

### 3. Explore the data

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 4. Train (on cloud GPU)

```bash
python src/train.py --config configs/train_config.yaml
```

## Results

| Model | IoU | F1 |
|-------|-----|----|
| VV Threshold | — | — |
| Random Forest | — | — |
| U-Net (ResNet-34) | — | — |

## References

- [SEN1Floods11 paper](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/papers/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2021_paper.pdf) (CVPR 2021)
- [Radiant MLHub](https://mlhub.earth/data/sen1floods11_v1.1)
