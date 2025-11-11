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


def load_dinov3_local(name: str, weights_path: str):
    key = _normalize_name(name)
    model = timm.create_model(key, pretrained=False)
    state_dict = _load_checkpoint(weights_path)
    ref_state = model.state_dict()
    state_dict = _maybe_strip_module_prefix(state_dict, list(ref_state.keys()))

    missing = [k for k in ref_state.keys() if k not in state_dict]
    unexpected = [k for k in state_dict.keys() if k not in ref_state]
    shape_mismatch = [
        (k, tuple(state_dict[k].shape), tuple(ref_state[k].shape))
        for k in state_dict.keys()
        if k in ref_state and getattr(state_dict[k], "shape", None) != getattr(ref_state[k], "shape", None)
    ]
    if missing or unexpected or shape_mismatch:
        # why: only show first 10 for readability
        details = {
            "model": key,
            "path": str(Path(weights_path).expanduser().resolve()),
            "missing": missing[:10],
            "unexpected": unexpected[:10],
            "shape_mismatch": shape_mismatch[:10],
        }
        raise RuntimeError(f"state_dict mismatch: {details}")
    model.load_state_dict(state_dict, strict=True)
    return model
