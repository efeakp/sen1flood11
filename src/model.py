"""U-Net model for flood extent segmentation."""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(
    encoder: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 2,
    num_classes: int = 1,
) -> nn.Module:
    """
    Returns a U-Net with the specified encoder.

    The first conv layer is adjusted for 2-channel SAR input (VV + VH).
    encoder_weights='imagenet' initialises the encoder from ImageNet weights,
    which still helps even with non-RGB input.
    """
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # raw logits; apply sigmoid in loss / inference
    )
    return model


def get_loss() -> nn.Module:
    # DiceLoss + BCE combined works well for imbalanced flood masks
    return smp.losses.DiceLoss(mode="binary") + smp.losses.SoftBCEWithLogitsLoss()
