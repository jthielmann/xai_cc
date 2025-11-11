from typing import Optional
import os
import torch
from difflib import get_close_matches

def _resolve_official_repo_dir(repo_dir: Optional[str] = None) -> str:
    cand = repo_dir or os.environ.get("DINOV3_REPO") or os.environ.get("DINOV3_LOCATION")
    if not cand:
        raise FileNotFoundError("Please set DINOV3_REPO or pass repo_dir='...'.")
    hub_path = os.path.join(cand, "hubconf.py")
    if not os.path.isfile(hub_path):
        raise FileNotFoundError(f"Official DINOv3 hubconf.py not found at: {hub_path}")
    this_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        if os.path.samefile(os.path.abspath(cand), this_dir):
            raise RuntimeError("repo_dir points to this wrapper; provide the official repo path.")
    except FileNotFoundError:
        pass
    return cand

def _load_state_dict(weights: Optional[str]) -> Optional[dict]:
    if not weights:
        return None
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    try:
        obj = torch.load(weights, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(weights, map_location="cpu")
    if isinstance(obj, dict):
        for k in ("model_ema", "model", "state_dict", "params"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    if not isinstance(obj, dict):
        raise RuntimeError("Unexpected checkpoint format.")
    return obj

def _build(entrypoint: str, weights: Optional[str] = None, *, repo_dir: Optional[str] = None, **kwargs):
    repo = _resolve_official_repo_dir(repo_dir)
    model = torch.hub.load(repo, entrypoint, source="local", weights=None, **kwargs)
    sd = _load_state_dict(weights)
    if sd is not None:
        model.load_state_dict(sd, strict=True)
    return model

def dinov3_vits16(weights: Optional[str] = None, **kwargs):
    return _build("dinov3_vits16", weights, **kwargs)

def dinov3_vits16plus(weights: Optional[str] = None, **kwargs):
    return _build("dinov3_vits16plus", weights, **kwargs)

def dinov3_vitb16(weights: Optional[str] = None, **kwargs):
    return _build("dinov3_vitb16", weights, **kwargs)

def dinov3_vitl16(weights: Optional[str] = None, **kwargs):
    return _build("dinov3_vitl16", weights, **kwargs)

def dinov3_vith16plus(weights: Optional[str] = None, **kwargs):
    return _build("dinov3_vith16plus", weights, **kwargs)

def dinov3_vit7b16(weights: Optional[str] = None, **kwargs):
    return _build("dinov3_vit7b16", weights, **kwargs)

def dinov3_convnext_tiny(weights: Optional[str] = None, **kwargs):
    return _build("dinov3_convnext_tiny", weights, **kwargs)

def dinov3_convnext_small(weights: Optional[str] = None, **kwargs):
    return _build("dinov3_convnext_small", weights, **kwargs)

def dinov3_convnext_base(weights: Optional[str] = None, **kwargs):
    return _build("dinov3_convnext_base", weights, **kwargs)

def dinov3_convnext_large(weights: Optional[str] = None, **kwargs):
    return _build("dinov3_convnext_large", weights, **kwargs)

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
        raise RuntimeError(f"Ambiguous callable '{raw}'. Possible matches: {', '.join(prefix_matches)}")
    substr_matches = [k for k in __all__ if name in k]
    if len(substr_matches) == 1:
        return substr_matches[0]
    if len(substr_matches) > 1:
        raise RuntimeError(f"Ambiguous callable '{raw}'. Possible matches: {', '.join(substr_matches)}")
    suggestions = get_close_matches(name, __all__, n=5, cutoff=0.5)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise AttributeError(f"Cannot find callable '{raw}' in hubconf.{hint}")

def __getattr__(name: str):
    resolved = _resolve_callable_name(name)
    return _EXPORTS[resolved]

def available_models():
    return list(__all__)
