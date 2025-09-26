import importlib
from typing import Dict, Optional

import torch

# v2 if available
v2_ready = importlib.util.find_spec("torchvision.transforms.v2") is not None
if v2_ready:
    from torchvision.transforms import v2 as T
else:
    import torchvision.transforms as T  # type: ignore

from script.data_processing.transforms import build_transforms as _build_transforms


def get_train_transforms(image_size: int = 256, frozen_encoder: bool = False):
    # Backward-compatible simple ImageNet-normalized train transforms
    return T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(image_size, scale=(0.75, 1.0), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 256):
    # Backward-compatible simple ImageNet-normalized eval transforms
    return T.Compose([
        T.ToImage(),
        T.Resize((image_size, image_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_transforms(cfg: Optional[Dict] = None, *, split: str = "train", normalize: bool = True):
    """Compatibility shim used across the repo.

    - If cfg is provided, use centralized builder with encoder-based stats.
    - If cfg is None, return ImageNet-normalized transforms (train/eval per split).
    - The 'normalize' flag is retained for callers that temporarily want raw tensors.
    """
    if cfg is None:
        base = get_train_transforms() if split == "train" else get_eval_transforms()
        if normalize:
            return base
        ops = getattr(base, "transforms", None)
        if ops is None:
            return base
        ops_wo_norm = [op for op in ops if not isinstance(op, T.Normalize)]
        return T.Compose(ops_wo_norm)

    # Build using centralized logic
    tfs = _build_transforms(cfg)
    tf = tfs["train" if split == "train" else "eval"]
    if normalize:
        return tf

    # Remove Normalize if requested (best-effort)
    ops = getattr(tf, "transforms", None)
    if ops is None:
        return tf
    ops_wo_norm = [op for op in ops if not isinstance(op, T.Normalize)]
    return T.Compose(ops_wo_norm)


def get_transforms_dinov3(resize_size: int = 256):
    to_tensor = T.ToImage()
    resize = T.Resize((resize_size, resize_size), antialias=True)
    to_float = T.ToDtype(torch.float32, scale=True)
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return T.Compose([to_tensor, resize, to_float, normalize])
