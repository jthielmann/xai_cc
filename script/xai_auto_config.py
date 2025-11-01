from __future__ import annotations
import os
import re
from typing import Any, Dict
from script.main_utils import parse_yaml_config


def build_auto_xai_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    defaults = parse_yaml_config("../sweeps/configs/eval_defaults")

    # Merge with user cfg (cfg wins)
    out: Dict[str, Any] = {**defaults, **cfg}

    # Set defaults without clobbering user values
    out.setdefault("diff", True)
    out.setdefault("scatter", True)
    out.setdefault("forward_to_csv", True)
    out.setdefault("umap", True)
    out.setdefault("crp", False)
    out.setdefault("lxt", False)
    out.setdefault("lrp", False)
    out.setdefault("pcx", False)

    enc_type = str(out.get("encoder_type", "")).lower()
    model_path = str(out.get("model_state_path", ""))
    model_config: Dict[str, Any] = {}

    if model_path:
        # try several common config names adjacent to the checkpoint directory
        for name in ("config.yaml", "config.yml", "config"):
            p = os.path.join(model_path, name)
            if os.path.exists(p):
                try:
                    model_config = parse_yaml_config(p)
                except Exception:
                    model_config = {}
                break

    enc_from_cfg = str(model_config.get("encoder_type", "")).lower()
    tokens = " ".join([enc_type, enc_from_cfg, model_path.lower()])

    # More robust matching
    is_resnet = "resnet" in tokens
    is_vit = "vit" in tokens

    # Only infer if user didn't pin these explicitly
    if "lxt" not in cfg and is_vit:
        out["lxt"] = True
    if is_resnet:
        out["lrp"] = True
        out["pcx"] = True

    return out
