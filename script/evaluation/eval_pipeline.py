import os
import torch
from typing import Any, Dict, Optional

from script.evaluation.crp_plotting import plot_crp, plot_crp_zennit, plot_crp2
from script.evaluation.lrp_plotting import plot_lrp, plot_lrp_custom
from script.evaluation.pcx_plotting import plot_pcx
from script.evaluation.tri_plotting import plot_triptych_from_merge
from script.evaluation.scatter_plotting import plot_scatter
from script.model.model_factory import get_encoder
from script.model.lit_model import load_model
from script.data_processing.data_loader import get_dataset_from_config
from script.data_processing.image_transforms import get_transforms
from torch.utils.data import DataLoader, Subset

class EvalPipeline:
    def __init__(self, config, run):
        self.config = config
        self.wandb_run = run
        self.model_src = self.config.get("model_config_path")
        self.model = self._load_model()

    def _load_state_dict_from_path(self, path: str) -> Dict[str, Any]:
        state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError(f"State dict file {path!r} did not contain a dictionary")
        return state

    def _normalize_state_dicts(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            raw = raw["state_dict"]
        if any(k in raw for k in ("encoder", "gene_heads", "sae")):
            return {k: raw[k] for k in ("encoder", "gene_heads", "sae") if k in raw}
        return {"encoder": raw}

    def _collect_state_dicts(self) -> Optional[Dict[str, Any]]:
        if self.config.get("encoder_state_path", None) and self.config.get("model_state_path", None):
            raise RuntimeError(
                f"corrupted config: encoder_state_path \n{self.config.get('encoder_state_path')}"
                f"\n{self.config.get('model_state_path')}"
            )

        if self.config.get("encoder_state_path", None) and self.config.get("gene_head_state_path", None):
            paths = {
                "encoder": self.config.get("encoder_state_path"),
                "gene_heads": self.config.get("gene_head_state_path"),
                "sae": self.config.get("sae_state_path"),
            }
            loaded = {k: self._load_state_dict_from_path("../models" + p) for k, p in paths.items() if p}
            if loaded:
                return loaded
        elif self.config.get("model_state_path") and not self.config.get("encoder_state_path", None):
            path = self.config.get("model_state_path") + "/best_model.pth"
            bundled = self._load_state_dict_from_path(path)
            return self._normalize_state_dicts(bundled)
        else:
            raise RuntimeError(
                f"corrupted config:\n"
                f"encoder_state_path \n{self.config.get('encoder_state_path', 'None')}"
                f"gene_head_state_path \n{self.config.get('gene_head_state_path', 'None')}"
                f"model_state_path \n{self.config.get('model_state_path', 'None')}"
            )

    def _load_model(self):
        state_dicts = self._collect_state_dicts()
        return load_model(self.config["model_config"], state_dicts)

    def run(self):
        if self.config.get("lrp"):
            # Select backend
            lrp_backend = str(self.config.get("lrp_backend", "zennit")).lower()
            # Load dataset like in training using config's dataset string
            eval_tf = get_transforms(self.config["model_config"], split="eval")
            debug = bool(self.config.get("debug", False))
            ds = get_dataset_from_config(
                dataset_name=self.config["model_config"]["dataset"],
                genes=None,
                split="val",
                debug=debug,
                transforms=eval_tf,
                samples=None,
                only_inputs=True,
                meta_data_dir=self.config["model_config"]["meta_data_dir"],
                gene_data_filename=self.config["model_config"]["gene_data_filename"]
            )
            n = min(10, len(ds))
            loader = DataLoader(Subset(ds, list(range(n))), batch_size=1, shuffle=False)
            if lrp_backend == "custom":
                plot_lrp_custom(self.model, loader, run=self.wandb_run)
            else:
                plot_lrp(self.model, loader, run=self.wandb_run)

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
                plot_crp2(self.model, ds_subset, run=self.wandb_run)
            else:
                plot_crp_zennit(self.model, ds_subset, run=self.wandb_run, max_items=n)

        if self.config.get("pcx"):
            pcx_backend = str(self.config.get("pcx_backend", "zennit")).lower()
            # For now, PCX uses the zennit/CRP-based pipeline in cluster_functions.
            # The backend flag is accepted for parity and future extension.
            plot_pcx(self.model, self.config, run=self.wandb_run)
        if self.config.get("diff"):
            plot_triptych_from_merge(
                self.config["data_dir"],
                self.config["patient"],
                self.config["gene"],
                self.config["out_path"],
                is_online=bool(self.wandb_run),
                wandb_run=self.wandb_run,
            )
        if self.config.get("scatter"):
            plot_scatter(
                self.config,
                self.model,
                wandb_run=self.wandb_run
            )
