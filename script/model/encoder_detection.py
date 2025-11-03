from __future__ import annotations

import torch.nn as nn

# Strict, top-level imports (no try/except, no fallbacks)
from timm.models.vision_transformer import VisionTransformer
from timm.models.convnext import ConvNeXt
from torchvision.models.resnet import ResNet


def detect_encoder_family(encoder: nn.Module) -> str:
    """Return a coarse family name for the given encoder module.

    Families:
    - 'vit'      — timm VisionTransformer and UNI-like ViTs
    - 'resnet'   — torchvision ResNet backbones
    - 'convnext' — timm ConvNeXt backbones
    - 'other'    — anything else
    """
    if isinstance(encoder, VisionTransformer):
        return "vit"
    if isinstance(encoder, ResNet):
        return "resnet"
    if isinstance(encoder, ConvNeXt):
        return "convnext"
    return "other"

