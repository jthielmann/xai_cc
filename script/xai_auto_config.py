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
    out.setdefault("diff", False)
    out.setdefault("scatter", False)
    out.setdefault("forward_to_csv", False)
    out.setdefault("forward_to_csv_simple", False)
    out.setdefault("umap", True)
    out.setdefault("crp", False)
    out.setdefault("lxt", False)
    out.setdefault("lrp", False)
    out.setdefault("pcx", False)

    enc_type = str(out.get("encoder_type", "")).lower()
    model_path = str(out.get("model_state_path", ""))


    name = "config"
    p = os.path.join(model_path, name)
    model_config = parse_yaml_config(p)

    enc_from_cfg = model_config["encoder_type"]
    tokens = " ".join([enc_type, enc_from_cfg, model_path.lower()])

    is_resnet = "resnet" in tokens
    is_vit = "vit" in tokens

    if "lxt" not in cfg and is_vit:
        out["lxt"] = True
    if is_resnet:
        out["lrp"] = True
        out["pcx"] = True

    return out
