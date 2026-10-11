# Flood Monitoring with SEN1Floods11

Machine learning methods to monitor the **extent and frequency of flooding** using Sentinel-1 SAR imagery from the [SEN1Floods11](https://github.com/cloudtostreet/SEN1Floods11) benchmark dataset (CVPR 2021).

## Overview

| | |
|---|---|
| **Task** | Binary semantic segmentation — flooded vs. non-flooded pixels |
| **Input** | 2-channel Sentinel-1 SAR (VV + VH polarisation, dB) |
| **Output** | Per-image flood extent masks + multi-event flood frequency maps |
| **Dataset** | 446 hand-labeled 512×512 chips across 11 flood events on 5 continents |
| **Split** | 252 train / 89 val / 90 test / 15 Bolivia (held-out) |
| **Class balance** | ~13.7% flood pixels (imbalanced — Dice+BCE loss used) |

## Project Structure

```
sen1flood11/
├── data/
│   ├── raw/                          # downloaded chips (not tracked by git)
│   │   ├── S1Hand/                   # Sentinel-1 SAR chips (VV + VH)
│   │   ├── LabelHand/                # binary flood masks
│   │   ├── S1OtsuLabelHand/          # Otsu threshold labels (baseline)
│   │   ├── S2Hand/                   # Sentinel-2 optical (reference)
│   │   ├── JRCWaterHand/             # JRC permanent water layer
│   │   └── flood_*_data.csv          # train / val / test / bolivia splits
│   └── processed/                    # model outputs and frequency maps
├── src/
│   ├── dataset.py      # PyTorch Dataset (loads VV/VH SAR + flood masks)
│   ├── model.py        # U-Net with ResNet-34 encoder
│   ├── train.py        # training loop with early stopping
│   ├── evaluate.py     # IoU and F1 metrics
│   └── frequency.py    # aggregate predictions → flood frequency map
├── notebooks/
│   └── 01_eda.ipynb    # exploratory data analysis
├── scripts/
│   ├── download_data.py  # fetch dataset via gsutil or Hugging Face
│   └── preprocess.py     # validate chips and compute dataset statistics
└── configs/
    └── train_config.yaml
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

**Option A — Google Cloud Storage (recommended):**
```bash
# Hand-labeled subset only (~700 MB, what you need for training)
gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled data/raw
gsutil -m cp "gs://sen1floods11/v1.1/splits/flood_handlabeled/*.csv" data/raw/
```

**Option B — via download script (full dataset ~14 GB):**
```bash
python scripts/download_data.py --method gsutil
```

**Option C — Hugging Face (no account needed):**
```bash
pip install datasets
python scripts/download_data.py --method hf
```

### 3. Validate the download

```bash
python scripts/preprocess.py --data data/raw
```

Expected output:
```
S1 chips:    446
Label chips: 446
VV backscatter mean:  -9.76 dB
VH backscatter mean:  -16.39 dB
Flood pixel fraction: 0.137
```

### 4. Explore the data

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 5. Train (on cloud GPU — Google Colab / Kaggle recommended)

```bash
python src/train.py --config configs/train_config.yaml
```

## Dataset Details

| Region | Chips | Notes |
|---|---|---|
| Ghana | ~72 | |
| India | ~54 | |
| USA | ~50 | |
| Pakistan | ~48 | |
| Mekong | ~42 | |
| Paraguay | ~40 | |
| Nigeria | ~38 | |
| Sri-Lanka | ~32 | |
| Spain | ~30 | |
| Somalia | ~25 | |
| Bolivia | 15 | **held-out** — geographic generalisation test |

## Model Results

| Model | Val IoU | Val F1 | Notes |
|---|---|---|---|
| VV Threshold | — | — | baseline |
| Random Forest | — | — | per-pixel features |
| U-Net (ResNet-34) | — | — | trained on cloud GPU |

## Flood Frequency Maps

After training, predictions across all events are aggregated to produce spatial flood frequency maps — each pixel value represents the fraction of events in which that location was classified as flooded. Useful for flood risk mapping and infrastructure planning.

## References

- [Sen1Floods11 paper](https://openaccess.thecvf.com/content/CVPRW2021/papers/w6/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2021_paper.pdf) — Bonafilia et al., CVPR 2021
- [Dataset on GCS](https://console.cloud.google.com/storage/browser/sen1floods11)
- [cloudtostreet/Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)
