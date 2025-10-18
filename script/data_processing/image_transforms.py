import importlib.util
from typing import Optional, Dict

import torch
from script.data_processing.transforms import build_transforms as _build_transforms
from script.configs.normalization import IMAGENET_MEAN, IMAGENET_STD

v2_ready = importlib.util.find_spec("torchvision.transforms.v2") is not None
if v2_ready:
    from torchvision.transforms import v2 as T
else:
    import torchvision.transforms as T  # type: ignore


def _resize(image_size: int):
    try:
        return T.Resize((image_size, image_size), antialias=True)
    except TypeError:
        return T.Resize((image_size, image_size))

def _rand_resized_crop(image_size: int):
    try:
        return T.RandomResizedCrop(image_size, scale=(0.75, 1.0), antialias=True)
    except TypeError:
        return T.RandomResizedCrop(image_size, scale=(0.75, 1.0))

def _to_image():
    return T.ToImage() if v2_ready else (lambda x: x)  # no-op on classic API

def _to_float_scaled():
    return T.ToDtype(torch.float32, scale=True) if v2_ready else T.ToTensor()


def get_train_transforms(image_size: int = 256, frozen_encoder: bool = False):
    return T.Compose([
        _to_image(),
        _rand_resized_crop(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        _to_float_scaled(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_eval_transforms(image_size: int = 256):
    if v2_ready:
        return T.Compose([
            T.ToImage(),
            _resize(image_size),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            _resize(image_size),
            T.ToTensor(),  # scales to [0,1]
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_transforms(cfg: Optional[Dict] = None, *, split: str = "train", normalize: bool = True):
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
