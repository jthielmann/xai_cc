# hubconf.py
"""
Minimal Torch Hub entry points for DINOv3-like ViT and ConvNeXt models.
- No satellite variants
- No HTTP download support (local file or None only)
- Supports unique-abbreviation lookup for callable names (e.g., "dinov3_vitb",
  "dinov3_vits16_", or "vitb16" if unique among exports)

Usage:
    import torch, os
    REPO_DIR = "/abs/path/to/this/folder"  # must contain this hubconf.py
    CKPT = "/abs/path/to/checkpoints/dinov3_vitb16.pth"  # or None

    # exact
    model = torch.hub.load(REPO_DIR, "dinov3_vitb16", source="local", weights=CKPT)
    # unique prefix
    model = torch.hub.load(REPO_DIR, "dinov3_vitb", source="local", weights=CKPT)
    # trailing underscore ok
    model = torch.hub.load(REPO_DIR, "dinov3_vits16_", source="local", weights=None)
"""

from typing import Optional, Tuple, Dict
import os
import torch
import timm
from difflib import get_close_matches

# -----------------------
# Internals
# -----------------------

def _load_local_state_dict(weights: Optional[str]):
    """Load a local checkpoint path into a state_dict. Returns None if weights is falsy."""
    if not weights:
        return None
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    obj = torch.load(weights, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        for k in ("model_ema", "model", "state_dict", "params"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    # Fallback: assume it's already a state_dict
    return obj

def _build_timm_backbone(model_name: str, **kwargs):
    """
    Create a TIMM backbone that returns features (not logits).
    'img_size' may be passed in via kwargs if the TIMM variant needs it (e.g., 384 for some '+').
    """
    return timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        global_pool="",
        **kwargs,
    )

def _load_model(timm_name: str, weights: Optional[str], *, img_size: int = 224, **kwargs):
    # Pass img_size only to non-ConvNeXt models
    if "convnext" in timm_name:
        kwargs.pop("img_size", None)
    else:
        kwargs["img_size"] = img_size
    model = _build_timm_backbone(timm_name, **kwargs)
    sd = _load_local_state_dict(weights)
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[hubconf] loaded weights into {timm_name} | missing={len(missing)}, unexpected={len(unexpected)}")
    return model

# -----------------------
# Name mappings
# -----------------------

_VIT_MAP: Dict[str, Tuple[str, int]] = {
    # DINOv3 ViT (web images)
    "dinov3_vits16": ("vit_small_patch16_224", 224),
    "dinov3_vits16plus": ("vit_small_patch16_384", 384),  # "+": 384-res variant
    "dinov3_vitb16": ("vit_base_patch16_224", 224),
    "dinov3_vitl16": ("vit_large_patch16_224", 224),
    # Best-effort maps for larger variants (require recent timm)
    "dinov3_vith16plus": ("vit_huge_patch14_224", 224),
    "dinov3_vit7b16": ("vit_giant_patch14_224", 224),
}

_CONVNEXT_MAP: Dict[str, str] = {
    "dinov3_convnext_tiny": "convnext_tiny",
    "dinov3_convnext_small": "convnext_small",
    "dinov3_convnext_base": "convnext_base",
    "dinov3_convnext_large": "convnext_large",
}

# -----------------------
# Exported Hub functions
# -----------------------
# Each accepts: weights=<LOCAL PATH or None> and forwards **kwargs to timm.create_model

def dinov3_vits16(weights: Optional[str] = None, **kwargs):
    name, img = _VIT_MAP["dinov3_vits16"]
    return _load_model(name, weights, img_size=img, **kwargs)

def dinov3_vits16plus(weights: Optional[str] = None, **kwargs):
    name, img = _VIT_MAP["dinov3_vits16plus"]
    return _load_model(name, weights, img_size=img, **kwargs)

def dinov3_vitb16(weights: Optional[str] = None, **kwargs):
    name, img = _VIT_MAP["dinov3_vitb16"]
    return _load_model(name, weights, img_size=img, **kwargs)

def dinov3_vitl16(weights: Optional[str] = None, **kwargs):
    name, img = _VIT_MAP["dinov3_vitl16"]
    return _load_model(name, weights, img_size=img, **kwargs)

def dinov3_vith16plus(weights: Optional[str] = None, **kwargs):
    name, img = _VIT_MAP["dinov3_vith16plus"]
    kwargs['patch_size'] = 16
    return _load_model(name, weights, img_size=img, **kwargs)

def dinov3_vit7b16(weights: Optional[str] = None, **kwargs):
    name, img = _VIT_MAP["dinov3_vit7b16"]
    return _load_model(name, weights, img_size=img, **kwargs)

def dinov3_convnext_tiny(weights: Optional[str] = None, **kwargs):
    return _load_model(_CONVNEXT_MAP["dinov3_convnext_tiny"], weights, **kwargs)

def dinov3_convnext_small(weights: Optional[str] = None, **kwargs):
    return _load_model(_CONVNEXT_MAP["dinov3_convnext_small"], weights, **kwargs)

def dinov3_convnext_base(weights: Optional[str] = None, **kwargs):
    return _load_model(_CONVNEXT_MAP["dinov3_convnext_base"], weights, **kwargs)

def dinov3_convnext_large(weights: Optional[str] = None, **kwargs):
    return _load_model(_CONVNEXT_MAP["dinov3_convnext_large"], weights, **kwargs)

# -----------------------
# Abbreviation support
# -----------------------

# Registry of exported callables
_EXPORTS = {
    "dinov3_vits16": dinov3_vits16,
    "dinov3_vits16plus": dinov3_vits16plus,
    "dinov3_vitb16": dinov3_vitb16,
    "dinov3_vitl16": dinov3_vitl16,
    "dinov3_vith16plus": dinov3_vith16plus,
    "dinov3_vit7b16": dinov3_vit7b16,
    "dinov3_convnext_tiny": dinov3_convnext_tiny,
    "dinov3_convnext_small": dinov3_convnext_small,
    "dinov3_convnext_base": dinov3_convnext_base,
    "dinov3_convnext_large": dinov3_convnext_large,
}
__all__ = sorted(_EXPORTS.keys())

def _resolve_callable_name(name: str) -> str:
    """
    Resolve a possibly abbreviated callable name to a unique exported name.

    Strategy:
      1) Strip trailing underscores (common typo when autocompleting).
      2) Unique prefix match against __all__.
      3) If no prefix matches, unique *substring* match against __all__.
      4) Otherwise raise with a helpful message.
    """
    if not isinstance(name, str):
        raise AttributeError(f"Invalid callable name type: {type(name)}")

    raw = name
    name = name.rstrip("_").lower()

    if name in _EXPORTS:
        return name

    prefix_matches = [k for k in __all__ if k.startswith(name)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        raise RuntimeError(
            f"Ambiguous callable '{raw}'. Possible matches: {', '.join(prefix_matches)}"
        )

    substr_matches = [k for k in __all__ if name in k]
    if len(substr_matches) == 1:
        return substr_matches[0]
    if len(substr_matches) > 1:
        raise RuntimeError(
            f"Ambiguous callable '{raw}'. Possible matches: {', '.join(substr_matches)}"
        )

    # Suggest close names for convenience
    suggestions = get_close_matches(name, __all__, n=5, cutoff=0.5)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise AttributeError(f"Cannot find callable '{raw}' in hubconf.{hint}")

def __getattr__(name: str):
    """
    Module-level fallback to support unique abbreviations when torch.hub looks up the callable.
    This is triggered when the attribute is not found by normal lookup.
    """
    resolved = _resolve_callable_name(name)
    return _EXPORTS[resolved]

def available_models():
    """Return the list of available exported callables."""
    return list(__all__)
