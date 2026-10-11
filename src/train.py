"""Training script for flood extent U-Net."""

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SEN1FloodsDataset
from evaluate import evaluate_batch
from model import build_model, get_loss


def train(config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    train_ds = SEN1FloodsDataset(
        split_csv=config["data"]["split_file"],
        data_dir=config["data"]["raw_dir"],
        split="train",
    )
    val_ds = SEN1FloodsDataset(
        split_csv=config["data"]["split_file"],
        data_dir=config["data"]["raw_dir"],
        split="val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    # --- Model ---
    model = build_model(
        encoder=config["model"]["encoder"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=config["data"]["in_channels"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    criterion = get_loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )

    checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_iou = 0.0
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]

    for epoch in range(1, config["training"]["epochs"] + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device).unsqueeze(1).float()

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validate ---
        model.eval()
        val_iou, val_f1 = 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} val  "):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device).unsqueeze(1).float()
                logits = model(images)
                metrics = evaluate_batch(logits, masks)
                val_iou += metrics["iou"]
                val_f1 += metrics["f1"]

        n_val = len(val_loader)
        val_iou /= n_val
        val_f1 /= n_val
        train_loss /= len(train_loader)

        scheduler.step()

        print(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
            f"val_iou={val_iou:.4f} | val_f1={val_f1:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            print(f"  Saved best model (IoU={best_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
