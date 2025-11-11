from pathlib import Path
from typing import Dict, List

import timm
import torch


_ALLOWED: List[str] = [
    "dinov3_vits16",
    "dinov3_vits16plus",
    "dinov3_vitb16",
    "dinov3_vitl16",
    "dinov3_vith16plus",
    "dinov3_vit7b16",
    "dinov3_convnext_tiny",
    "dinov3_convnext_small",
    "dinov3_convnext_base",
    "dinov3_convnext_large",
]


def available_dinov3_models() -> List[str]:
    return list(_ALLOWED)


def _normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(f"invalid name: {name}")
    key = name.strip().lower()
    if key not in _ALLOWED:
        raise ValueError(
            f"unknown model '{name}'. expected one of: {', '.join(_ALLOWED)}"
        )
    return key


def _load_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint not found: {resolved}")
    obj = torch.load(str(resolved), map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError(f"unexpected checkpoint type {type(obj)} at {resolved}")
    candidates = [
        k
        for k in ("model_ema", "model", "state_dict", "params")
        if k in obj and isinstance(obj[k], dict)
    ]
    # why: strict; multiple candidates ambiguous; require single
    if len(candidates) == 1:
        return obj[candidates[0]]
    if len(candidates) == 0:
        return obj
    raise RuntimeError(f"multiple state_dict keys {candidates} in {resolved}")


def _maybe_strip_module_prefix(
    state_dict: Dict[str, torch.Tensor], ref_keys: List[str]
) -> Dict[str, torch.Tensor]:
    # strip only if all keys prefixed and keys match.
    if all(k.startswith("module.") for k in state_dict.keys()):
        stripped = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        if set(stripped.keys()) == set(ref_keys):
            return stripped
    return state_dict


# NEW: resolve a .pth path from either a file or a directory (e.g., "../encoders")
def _resolve_weights_path_for_name(name: str, weights_path: str) -> str:
    p = Path(weights_path).expanduser().resolve()
    if p.is_file():
        return str(p)
    if p.is_dir():
        # Prefer official pretrain files: dinov3_<name>_pretrain_*.pth
        pattern_primary = f"{name}_pretrain_*.pth"
        pattern_fallback = f"{name}_*.pth"
        candidates = sorted(str(x) for x in p.glob(pattern_primary))
        if not candidates:
            candidates = sorted(str(x) for x in p.glob(pattern_fallback))
        if not candidates:
            raise FileNotFoundError(f"no checkpoint for '{name}' found in directory: {p}")
        if len(candidates) > 1:
            raise RuntimeError(
                f"ambiguous checkpoints for '{name}' in {p}: "
                f"{[Path(c).name for c in candidates]}"
            )
        return candidates[0]
    raise FileNotFoundError(f"weights_path not found: {p}")


def load_dinov3_local(name: str, weights_path: str):
    key = _normalize_name(name)
    resolved_weights = _resolve_weights_path_for_name(key, weights_path)

    # Load official Meta checkpoint via Torch Hub (GitHub). This uses the repo's hubconf.
    model = torch.hub.load(
        'facebookresearch/dinov3:main',
        key,
        source='github',
        weights=resolved_weights,
    )
    return model
