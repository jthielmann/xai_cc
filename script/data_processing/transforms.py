import importlib
from typing import Dict

import torch

# Prefer torchvision v2; fall back to v1 if unavailable
v2_ready = importlib.util.find_spec("torchvision.transforms.v2") is not None
if v2_ready:
    from torchvision.transforms import v2 as T
else:
    import torchvision.transforms as T  # type: ignore

from script.configs.normalization import resolve_norm
from script.data_processing.custom_transforms import Occlude


def _assert_single_normalize(seq) -> None:
    """Best-effort guard: ensure exactly one Normalize op in a composed pipeline."""
    try:
        # v2 Compose exposes .transforms, old Compose also does
        ops = getattr(seq, "transforms", None) or []
        n = sum(1 for t in ops if isinstance(t, T.Normalize))
        assert n == 1, f"Expected exactly one Normalize, found {n}"
    except Exception:
        # Non-fatal: only guard in common cases
        pass


def build_transforms(cfg: Dict) -> Dict[str, T.Compose]:
    """Build train/eval transforms using encoder-based normalization.

    Returns a dict with keys: 'train' and 'eval'.
    """
    encoder_type = cfg.get("encoder_type", "resnet50imagenet")
    image_size = int(cfg.get("image_size", 256))
    frozen_encoder = bool(cfg.get("freeze_encoder", False))

    stats = resolve_norm(encoder_type)

    # Geometric and color augs for training
    train_ops = [
        T.ToImage(),
        T.RandomResizedCrop(image_size, scale=(0.75, 1.0), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        # 0/90/180/270° rotations
        T.RandomChoice([
            T.RandomRotation([0, 0]),
            T.RandomRotation([90, 90]),
            T.RandomRotation([180, 180]),
            T.RandomRotation([270, 270]),
        ], p=[0.25, 0.25, 0.25, 0.25]),
        T.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.97, 1.03)),
    ]
    if not frozen_encoder:
        train_ops.append(T.RandomApply([T.ColorJitter(0.05, 0.05, 0.03, 0.01)], p=0.2))

    # Tensor + Occlude + Normalize
    train_ops += [
        T.ToDtype(torch.float32, scale=True),
        # Occlude after tensor conversion, before Normalize
        Occlude(patch_size_x=32, patch_size_y=32, patch_vary_width=8, patch_min_size=8, patch_max_size=64, use_batch=False),
        T.Normalize(mean=stats.mean, std=stats.std),
    ]
    train_tf = T.Compose(train_ops)

    eval_tf = T.Compose([
        T.ToImage(),
        T.Resize((image_size, image_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=stats.mean, std=stats.std),
    ])

    # Guards
    _assert_single_normalize(train_tf)
    _assert_single_normalize(eval_tf)

    return {"train": train_tf, "eval": eval_tf}

