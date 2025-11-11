# hubconf.py
"""
Notes:
  - Official DINOv3 Hub entrypoints: dinov3_vits16, dinov3_vits16plus, dinov3_vitb16,
    dinov3_vitl16, dinov3_vith16plus, dinov3_vit7b16, and ConvNeXt variants. They all
    accept `weights=<URL or local path>`. (See official README’s code snippet.)  # README cites lines 356-370
"""

from typing import Optional, Tuple, Dict
import os
import torch
from difflib import get_close_matches

# -----------------------
# Internals
# -----------------------

def _resolve_official_repo_dir(repo_dir: Optional[str] = None) -> str:
    """
    Where is the OFFICIAL facebookresearch/dinov3 repo cloned?
    Priority:
      1) explicit kwarg `repo_dir=...`
      2) env var DINOV3_REPO
      3) env var DINOV3_LOCATION (used in official notebooks)
    """
    cand = repo_dir or os.environ.get("DINOV3_REPO") or os.environ.get("DINOV3_LOCATION")
    if not cand:
        raise FileNotFoundError(
            "[hubconf] Please set DINOV3_REPO env var or pass repo_dir='...'"
        )
    hub_path = os.path.join(cand, "hubconf.py")
    if not os.path.isfile(hub_path):
        raise FileNotFoundError(
            f"[hubconf] Official DINOv3 hubconf.py not found at: {hub_path}"
        )
    # Avoid recursion if someone points to *this* directory by mistake.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        if os.path.samefile(os.path.abspath(cand), this_dir):
            raise RuntimeError(
                "[hubconf] repo_dir points to this wrapper, not the official repo. "
                "Set DINOV3_REPO to the facebookresearch/dinov3 clone."
            )
    except FileNotFoundError:
        pass
    return cand

def _load_local_state_dict(weights: Optional[str]):
    """
    Only used for auto-variant detection. Returns a plain state_dict or None.
    """
    if not weights:
        return None
    # If it's clearly a URL, skip local introspection
    if isinstance(weights, str) and (weights.startswith("http://") or weights.startswith("https://")):
        return None
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"[hubconf] Checkpoint not found: {weights}")
    obj = torch.load(weights, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        for k in ("model_ema", "model", "state_dict", "params"):
            if k in obj and isinstance(obj[k], dict):
                print(f"[hubconf] using subkey='{k}' from checkpoint for variant detection")
                return obj[k]
    return obj if isinstance(obj, dict) else None

def _infer_vit_variant_from_sd(sd: dict, weights_path: Optional[str]) -> Optional[str]:
    """
    Inspect ViT weights to select the correct official entrypoint.
    Returns one of:
      'dinov3_vits16', 'dinov3_vits16plus', 'dinov3_vitb16',
      'dinov3_vitl16', 'dinov3_vith16plus', 'dinov3_vit7b16'
    or None if not enough info.

    Heuristics:
      - embed dim from patch_embed.proj.weight (out_channels)
      - depth from number of Transformer blocks (max index + 1)
      - For S vs S+ (384px), prefer '+’ if the weight path hints 'plus'/'384'.
    """
    if not sd:
        return None

    # Detect ViT checkpoints: look for transformer/patch keys
    if not any(k.startswith("patch_embed.") for k in sd.keys()):
        return None
    if not any(k.startswith("blocks.") for k in sd.keys()):
        return None

    # embed dim from patch_embed
    pe = sd.get("patch_embed.proj.weight", None)
    if pe is None:
        # Some checkpoints may have 'patch_embed.proj.weight' under a different key.
        # Try qkv as a fallback (qkv: [3*embed, embed])
        qkv = next((v for k, v in sd.items() if k.endswith("attn.qkv.weight")), None)
        if qkv is not None and hasattr(qkv, "shape"):
            embed = int(qkv.shape[1])
        else:
            return None
    else:
        embed = int(pe.shape[0]) if hasattr(pe, "shape") else None
        if embed is None:
            return None

    # depth from largest blocks.<i>.*
    depth = -1
    for k in sd.keys():
        if k.startswith("blocks.") and ".attn.qkv.weight" in k:
            try:
                idx = int(k.split(".")[1])
                depth = max(depth, idx + 1)
            except Exception:
                pass
    if depth <= 0:
        return None

    # Map (embed, depth) to official entrypoint
    # S: 384×12 ; B: 768×12 ; L: 1024×24 ; H+: 1280×32 ; 7B: 4096×40
    # (The official "+"/H+ variants use the 384px pretraining recipe.)  # sizes from model card/README
    if embed == 384 and depth == 12:
        hint = (weights_path or "").lower()
        return "dinov3_vits16plus" if ("plus" in hint or "384" in hint) else "dinov3_vits16"
    if embed == 768 and depth == 12:
        return "dinov3_vitb16"
    if embed == 1024 and depth == 24:
        return "dinov3_vitl16"
    if embed == 1280 and depth == 32:
        return "dinov3_vith16plus"
    if embed == 4096 and depth == 40:
        return "dinov3_vit7b16"

    return None

def _load_official(entrypoint: str,
                   weights: Optional[str] = None,
                   *,
                   repo_dir: Optional[str] = None,
                   allow_auto_variant: bool = True,
                   **kwargs):
    """
    Build via official facebookresearch/dinov3 Hub entrypoints.

    If `allow_auto_variant` and a ViT checkpoint is provided, we override `entrypoint`
    with the one inferred from the checkpoint if they differ (to "use the matching architecture").
    """
    official_repo = _resolve_official_repo_dir(repo_dir)
    original_entry = entrypoint

    # Try auto-variant selection for ViT weights
    if allow_auto_variant and weights:
        sd = _load_local_state_dict(weights)
        inferred = _infer_vit_variant_from_sd(sd, weights) if sd else None
        if inferred and inferred != entrypoint:
            print(f"[hubconf] auto-selecting '{inferred}' (was '{entrypoint}') "
                  f"based on checkpoint shapes")
            entrypoint = inferred

    # Official Hub call (supports local path or URL in `weights`)
    # See official README for the list of entrypoints and the argument `weights=...`.
    # Example there:
    # dinov3_vitb16 = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local',
    #                                weights=<CHECKPOINT/URL/OR/PATH>)
    model = torch.hub.load(official_repo, entrypoint, source="local", weights=weights, **kwargs)

    if original_entry != entrypoint:
        print(f"[hubconf] NOTE: Requested '{original_entry}' but built '{entrypoint}' "
              f"to match the checkpoint.")

    return model

# -----------------------
# Exported Hub functions
# -----------------------
# All accept: weights=<LOCAL PATH or URL or None>; repo_dir=<path to official repo>; **kwargs forwarded.

def dinov3_vits16(weights: Optional[str] = None, **kwargs):
    return _load_official("dinov3_vits16", weights, **kwargs)

def dinov3_vits16plus(weights: Optional[str] = None, **kwargs):
    return _load_official("dinov3_vits16plus", weights, **kwargs)

def dinov3_vitb16(weights: Optional[str] = None, **kwargs):
    return _load_official("dinov3_vitb16", weights, **kwargs)

def dinov3_vitl16(weights: Optional[str] = None, **kwargs):
    return _load_official("dinov3_vitl16", weights, **kwargs)

def dinov3_vith16plus(weights: Optional[str] = None, **kwargs):
    return _load_official("dinov3_vith16plus", weights, **kwargs)

def dinov3_vit7b16(weights: Optional[str] = None, **kwargs):
    return _load_official("dinov3_vit7b16", weights, **kwargs)

def dinov3_convnext_tiny(weights: Optional[str] = None, **kwargs):
    return _load_official("dinov3_convnext_tiny", weights, **kwargs)

def dinov3_convnext_small(weights: Optional[str] = None, **kwargs):
    return _load_official("dinov3_convnext_small", weights, **kwargs)

def dinov3_convnext_base(weights: Optional[str] = None, **kwargs):
    return _load_official("dinov3_convnext_base", weights, **kwargs)

def dinov3_convnext_large(weights: Optional[str] = None, **kwargs):
    return _load_official("dinov3_convnext_large", weights, **kwargs)

# Convenience: auto-detect variant directly
def dinov3_auto(weights: Optional[str] = None, **kwargs):
    """
    Build the correct official DINOv3 ViT variant by inspecting `weights`.
    If `weights` is None or a URL we cannot introspect, raises a helpful error.
    """
    sd = _load_local_state_dict(weights)
    if not sd:
        raise RuntimeError(
            "[hubconf] dinov3_auto requires a local checkpoint path in `weights=` "
            "so it can infer the variant (embed dim & depth)."
        )
    variant = _infer_vit_variant_from_sd(sd, weights)
    if not variant:
        raise RuntimeError("[hubconf] Could not infer DINOv3 ViT variant from checkpoint.")
    print(f"[hubconf] dinov3_auto resolved to '{variant}'")
    return _load_official(variant, weights, **kwargs)

# -----------------------
# Abbreviation support
# -----------------------

_EXPORTS = {
    # ViTs
    "dinov3_vits16": dinov3_vits16,
    "dinov3_vits16plus": dinov3_vits16plus,
    "dinov3_vitb16": dinov3_vitb16,
    "dinov3_vitl16": dinov3_vitl16,
    "dinov3_vith16plus": dinov3_vith16plus,
    "dinov3_vit7b16": dinov3_vit7b16,
    # ConvNeXts
    "dinov3_convnext_tiny": dinov3_convnext_tiny,
    "dinov3_convnext_small": dinov3_convnext_small,
    "dinov3_convnext_base": dinov3_convnext_base,
    "dinov3_convnext_large": dinov3_convnext_large,
    # Auto
    "dinov3_auto": dinov3_auto,
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
    Triggered when the attribute is not found by normal lookup.
    """
    resolved = _resolve_callable_name(name)
    return _EXPORTS[resolved]

def available_models():
    """Return the list of available exported callables."""
    return list(__all__)
