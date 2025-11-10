import torch
from typing import Any, Dict


def auto_device(model: torch.nn.Module) -> torch.device:
    """Infer a sensible device from a model, with fallbacks.

    Returns the device of the first model parameter if available,
    otherwise chooses CUDA, then MPS, then CPU.
    """
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
                else "cpu"
            )
        )


def load_state_dict_from_path(path: str) -> Dict[str, Any]:
    """Load a state dict from a file path and validate type."""
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"State dict file {path!r} did not contain a dictionary")
    return state


def normalize_state_dicts(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize various checkpoint layouts to a unified dict.

    Supports either a raw state_dict mapping, a checkpoint with a top-level
    "state_dict" key, or a composite dict containing keys like
    "encoder", "gene_heads", and "sae".
    """
    if "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    if any(k in raw for k in ("encoder", "gene_heads", "sae")):
        return {k: raw[k] for k in ("encoder", "gene_heads", "sae") if k in raw}
    return {"encoder": raw}


def collect_state_dicts(config: Dict[str, Any]) -> Dict[str, Any]:
    """Collect and load model state dicts based on config.

    Handles either separate encoder/gene_head/sae paths or a bundled
    model path containing a best_model.pth.
    """
    if config.get("encoder_state_path", None) and config.get("model_state_path", None):
        raise RuntimeError(
            "corrupted config: encoder_state_path and model_state_path are both set\n"
            f"encoder_state_path: {config.get('encoder_state_path')}\n"
            f"model_state_path: {config.get('model_state_path')}"
        )

    if config.get("encoder_state_path", None) and config.get("gene_head_state_path", None):
        paths = {
            "encoder": config.get("encoder_state_path"),
            "gene_heads": config.get("gene_head_state_path"),
            "sae": config.get("sae_state_path"),
        }
        loaded = {
            k: load_state_dict_from_path("../models" + p) for k, p in paths.items() if p
        }
        if loaded:
            return loaded
    elif config.get("model_state_path") and not config.get("encoder_state_path", None):
        path = config.get("model_state_path") + "/best_model.pth"
        bundled = load_state_dict_from_path(path)
        return normalize_state_dicts(bundled)
    else:
        raise RuntimeError(
            "corrupted config:\n"
            f"encoder_state_path \n{config.get('encoder_state_path', 'None')}"
            f"gene_head_state_path \n{config.get('gene_head_state_path', 'None')}"
            f"model_state_path \n{config.get('model_state_path', 'None')}"
        )
