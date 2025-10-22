import itertools
import os
import torch
from typing import Any, Dict, Optional

from script.evaluation.crp_plotting import plot_crp_zennit, plot_crp
from script.model.lit_model import load_lit_regressor
from script.data_processing.data_loader import get_dataset_from_config
from script.data_processing.image_transforms import get_transforms
from torch.utils.data import DataLoader, Subset
from script.main_utils import prepare_cfg
import hashlib
import os
from script.evaluation.eval_helpers import (
    collect_state_dicts
)

class EvalPipeline:
    def __init__(self, config, run):
        self.config = prepare_cfg(config)
        self.wandb_run = run
        self.model_src = self.config.get("model_config_path")
        self.model = self._load_model()
        # Derive a model-based folder name and ensure base exists
        self.model_name = self._derive_model_name()
        base = os.path.join(self.config.get("eval_path", self.config.get("out_path", "./xai_out")), self.model_name)
        os.makedirs(base, exist_ok=True)

    def _load_model(self):
        state_dicts = collect_state_dicts(self.config)
        return load_lit_regressor(self.config["model_config"], state_dicts)

    def _derive_model_name(self) -> str:
        def _rel_model_path(p: str) -> str:
            s = str(p).replace("\\", "/")
            if "/models/" in s:
                s = s.split("/models/", 1)[1]
            elif s.startswith("../models/"):
                s = s[len("../models/"):]
            return s.strip("/")

        parts = []
        ms = self.config.get("model_state_path")
        enc = self.config.get("encoder_state_path")
        head = self.config.get("gene_head_state_path")
        if ms:
            parts.append(_rel_model_path(ms))
        else:
            if enc:
                parts.append(_rel_model_path(enc))
            if head:
                parts.append(_rel_model_path(head))

        name = os.path.join(*parts) if parts else "eval"
        MAX_LEN = 160
        if len(name) <= MAX_LEN:
            return name
        tokens = [t for t in name.split("/") if t]
        tail = "/".join(tokens[-2:]) if len(tokens) >= 2 else tokens[-1]
        h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        short = f"{tail}/__{h}"
        return short[:MAX_LEN]

    def run(self):
        if self.config.get("crp"):
            crp_backend = str(self.config.get("crp_backend", "zennit")).lower()
            eval_tf = get_transforms(self.config["model_config"], split="eval")
            ds = get_dataset_from_config(
                dataset_name=self.config["dataset"],
                genes=None,
                split="val",
                debug=bool(self.config.get("debug", False)),
                transforms=eval_tf,
                samples=None,
                only_inputs=False,
                meta_data_dir=self.config.get("meta_data_dir", "/meta_data/"),
                gene_data_filename=self.config.get("gene_data_filename", "gene_data.csv"),
            )
            n = min(10, len(ds))
            ds_subset = Subset(ds, list(range(n)))
            out_dir = os.path.join(self.config.get("eval_path", self.config.get("out_path", "./xai_out")), self.model_name, "crp")
            os.makedirs(out_dir, exist_ok=True)
            # Require explicit target_layer in config (no default). Allow special value 'encoder'.
            target_layer = self.config.get("target_layer")
            if not target_layer or not isinstance(target_layer, str):
                raise ValueError("CRP requires 'target_layer' in config (e.g., 'encoder' or a dot-path layer name).")
            # Validate target_layer exists on model
            if target_layer != "encoder":
                names = {n for n, _ in self.model.named_modules()}
                if target_layer not in names:
                    raise KeyError(f"target_layer '{target_layer}' not found in model named_modules().")

            if crp_backend == "custom":
                plot_crp(self.model, ds_subset, run=self.wandb_run, out_dir=out_dir, layer_name=target_layer)
            else:
                plot_crp_zennit(self.model, ds_subset, run=self.wandb_run, max_items=n, out_dir=out_dir, layer_name=target_layer)
