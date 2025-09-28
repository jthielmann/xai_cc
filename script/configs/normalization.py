from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class NormStats:
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


# Canonical ImageNet statistics
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# Map families/encoders to normalization stats
ENCODER_NORM: dict[str, NormStats] = {
    "resnet50imagenet": NormStats(IMAGENET_MEAN, IMAGENET_STD),
    "resnet50random":   NormStats(IMAGENET_MEAN, IMAGENET_STD),
    "dino":             NormStats(IMAGENET_MEAN, IMAGENET_STD),
    "dinov3":           NormStats(IMAGENET_MEAN, IMAGENET_STD),
    "uni":              NormStats(IMAGENET_MEAN, IMAGENET_STD),
}


def resolve_norm(encoder_type: str) -> NormStats:
    """Resolve normalization stats for a given encoder type.

    Falls back to ImageNet stats for known families and unknown types.
    """
    t = (encoder_type or "").strip().lower()
    # direct or prefix match
    for key in ENCODER_NORM:
        if t == key or t.startswith(key) or key in t:
            return ENCODER_NORM[key]
    # family-based fallbacks
    if t.startswith("resnet") or t.startswith("dinov3") or "uni" in t or "dino" in t:
        return ENCODER_NORM["resnet50imagenet"]
    # default
    return ENCODER_NORM["resnet50imagenet"]

