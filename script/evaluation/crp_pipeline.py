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
from script.evaluation.eval_helpers import (
    collect_state_dicts
)

class EvalPipeline:
    def __init__(self, config, run):
        self.config = prepare_cfg(config)
        self.wandb_run = run
        self.model_src = self.config.get("model_config_path")
        self.model = self._load_model()

    def _load_model(self):
        state_dicts = collect_state_dicts(self.config)
        return load_lit_regressor(self.config["model_config"], state_dicts)

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
            )
            n = min(10, len(ds))
            ds_subset = Subset(ds, list(range(n)))
            if crp_backend == "custom":
                plot_crp(self.model, ds_subset, run=self.wandb_run)
            else:
                plot_crp_zennit(self.model, ds_subset, run=self.wandb_run, max_items=n)