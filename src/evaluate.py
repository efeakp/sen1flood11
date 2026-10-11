"""Evaluation metrics for flood extent segmentation."""

import numpy as np
import torch


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Intersection over Union for binary masks."""
    pred_bin = (torch.sigmoid(pred) > threshold).bool()
    target_bin = target.bool()

    intersection = (pred_bin & target_bin).float().sum()
    union = (pred_bin | target_bin).float().sum()

    if union == 0:
        return 1.0
    return (intersection / union).item()


def f1_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """F1 score (Dice coefficient) for binary masks."""
    pred_bin = (torch.sigmoid(pred) > threshold).bool()
    target_bin = target.bool()

    tp = (pred_bin & target_bin).float().sum()
    fp = (pred_bin & ~target_bin).float().sum()
    fn = (~pred_bin & target_bin).float().sum()

    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0
    return (2 * tp / denom).item()


def evaluate_batch(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    return {
        "iou": iou_score(pred, target, threshold),
        "f1": f1_score(pred, target, threshold),
    }
