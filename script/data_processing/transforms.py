import importlib
from typing import Dict, List

import torch

# Prefer torchvision v2; fall back to v1 if unavailable
v2_ready = importlib.util.find_spec("torchvision.transforms.v2") is not None
if v2_ready:
    from torchvision.transforms import v2 as T
else:
    import torchvision.transforms as T  # type: ignore

from script.model.model_factory import get_encoder_transforms
from script.data_processing.custom_transforms import Occlude


def _assert_single_normalize(seq) -> None:
    """Ensure exactly one Normalize op in a composed pipeline."""
    ops = getattr(seq, "transforms", None)
    n = sum(1 for t in ops if isinstance(t, T.Normalize))
    assert n == 1, f"Expected exactly one Normalize, found {n}"

# Assumes your Option-A version:
# get_encoder_transforms(encoder_type, resize_size=224, extra=None, place="pre_norm")

def build_transforms(cfg: Dict) -> Dict[str, T.Compose]:
    """
    Build train/eval transforms using encoder-based normalization.
    Reads optional augs from cfg. Example keys (all optional):
      encoder_type: str (default 'resnet50imagenet')
      image_size: int (default 256)
      freeze_encoder: bool (default False)
      # geometry
      hflip: bool (default True)
      vflip: bool (default False)
      rot90s: bool (default True)
      affine: bool (default True)
      # color
      color_jitter: bool (default False; auto-disabled if freeze_encoder=True)
      color_jitter_params: tuple (brightness, contrast, saturation, hue)
      color_jitter_p: float
      # occlusion
      occlude: bool (default False)
      occ_patch_size_x / occ_patch_size_y / occ_patch_vary_width / occ_patch_min_size /
      occ_patch_max_size / occ_use_batch
      # cropping (note: applied after base resize; see comment below)
      rrc: bool (default False)
      rrc_scale: tuple (default (0.75, 1.0))
    """
    encoder_type = cfg["encoder_type"]
    image_size   = int(cfg.get("image_size", 256))
    frozen       = cfg["freeze_encoder"]

    # --- TRAIN EXTRAS (inserted pre-normalization) ---
    train_extra: List = []

    # Geometry
    if cfg.get("hflip", True):
        train_extra.append(T.RandomHorizontalFlip(p=0.5))
    # vflip default: False — vertical flips can break real-world semantics (orientation)
    # and add little beyond rot90s + hflip (which already cover all symmetries).
    # Enable only if your labels are orientation-invariant.
    if cfg.get("vflip", False):
        train_extra.append(T.RandomVerticalFlip(p=0.5))
    if cfg.get("rot90s", True):
        train_extra.append(T.RandomChoice([
            T.RandomRotation([0, 0]),
            T.RandomRotation([90, 90]),
            T.RandomRotation([180, 180]),
            T.RandomRotation([270, 270]),
        ], p=[0.25, 0.25, 0.25, 0.25]))
    if cfg.get("affine", True):
        train_extra.append(T.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.97, 1.03)))

    # Optional RandomResizedCrop (note: base pipeline already resizes;
    # applying RRC here works but reduces multi-scale effect vs doing it *before* resize)
    if cfg.get("rrc", False):
        rrc_scale = tuple(cfg.get("rrc_scale", (0.75, 1.0)))
        train_extra.append(T.RandomResizedCrop(image_size, scale=rrc_scale, antialias=True))

    # Color (skip if encoder is frozen, unless explicitly forced true)
    if cfg.get("color_jitter", False) and not frozen:
        b, c, s, h = cfg.get("color_jitter_params", (0.05, 0.05, 0.03, 0.01))
        cj_p = float(cfg.get("color_jitter_p", 0.2))
        train_extra.append(T.RandomApply([T.ColorJitter(b, c, s, h)], p=cj_p))

    # Occlusion-style aug (expects float tensor, so placing pre-norm is perfect)
    if cfg.get("occlude", False):
        train_extra.append(
            Occlude(
                patch_size_x=cfg.get("occ_patch_size_x", 32),
                patch_size_y=cfg.get("occ_patch_size_y", 32),
                patch_vary_width=cfg.get("occ_patch_vary_width", 8),
                patch_min_size=cfg.get("occ_patch_min_size", 8),
                patch_max_size=cfg.get("occ_patch_max_size", 64),
                use_batch=cfg.get("occ_use_batch", False),
            )
        )

    # Build final pipelines by hooking extras *before* Normalize
    train_tf = get_encoder_transforms(
        encoder_type=encoder_type,
        resize_size=image_size,
        extra=train_extra,
        place="pre_norm",
    )
    eval_tf = get_encoder_transforms(
        encoder_type=encoder_type,
        resize_size=image_size,
        extra=None,
        place="pre_norm",
    )

    # Optional guard: catch double-normalization mistakes in debug runs
    if cfg.get("debug", False):
        _assert_single_normalize(train_tf)
        _assert_single_normalize(eval_tf)

    return {"train": train_tf, "eval": eval_tf}
