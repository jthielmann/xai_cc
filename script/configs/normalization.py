from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class NormStats:
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


# Canonical ImageNet/DINO stats
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# Map known encoders to their expected input normalization stats
ENCODER_NORM: Dict[str, NormStats] = {
    # torchvision ResNet50 pretrained on ImageNet
    "resnet50imagenet": NormStats(IMAGENET_MEAN, IMAGENET_STD),
    # random init still uses ImageNet normalization for inputs by convention
    "resnet50random":   NormStats(IMAGENET_MEAN, IMAGENET_STD),
    # facebookresearch/dino resnet-50 hub model uses standard ImageNet stats
    "dino":             NormStats(IMAGENET_MEAN, IMAGENET_STD),
    # fallback for other ViT-based encoders unless specified otherwise
    "unimodel":         NormStats(IMAGENET_MEAN, IMAGENET_STD),
}


def resolve_norm(encoder_type: str) -> NormStats:
    """Return normalization stats for the given encoder type.

    Falls back to ImageNet stats if the encoder is unknown.
    """
    et = (encoder_type or "").lower()
    return ENCODER_NORM.get(et, NormStats(IMAGENET_MEAN, IMAGENET_STD))

