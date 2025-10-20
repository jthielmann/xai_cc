import os
from typing import Any, Dict, Optional

#from script.evaluation.crp_plotting import plot_crp, plot_crp_zennit, plot_crp2
from script.evaluation.lrp_plotting import plot_lrp, plot_lrp_custom
from script.evaluation.lxt_plotting import plot_lxt
from script.evaluation.pcx_plotting import plot_pcx
from script.evaluation.tri_plotting import plot_triptych_from_merge
from script.evaluation.scatter_plotting import plot_scatter
from script.evaluation.generate_results import generate_results
from script.model.model_factory import get_encoder
from script.model.lit_model import load_lit_regressor
from script.data_processing.data_loader import get_dataset_from_config, get_dataset
from script.data_processing.image_transforms import get_transforms, get_eval_transforms
from torch.utils.data import DataLoader, Subset
from script.main_utils import prepare_cfg
from script.evaluation.eval_helpers import (
    auto_device,
    collect_state_dicts,
)

class EvalPipeline:
    def __init__(self, config, run):
        self.config = prepare_cfg(config)
        self.wandb_run = run
        # Single source of truth for run_name: require it in config
        self.run_name = self.config.get("run_name")
        if not self.run_name:
            raise ValueError("Config must provide 'run_name' (single source of truth).")
        self.model_src = self.config.get("model_config_path")
        self.model = self._load_model()

    def _load_model(self):
        state_dicts = collect_state_dicts(self.config)
        return load_lit_regressor(self.config["model_config"], state_dicts)

    def run(self):
        if self.config.get("lrp"):
            lrp_backend = str(self.config.get("lrp_backend", "zennit")).lower()
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

        if self.config.get("pcx"):
            pcx_backend = str(self.config.get("pcx_backend", "zennit")).lower()
            # For now, PCX uses the zennit/CRP-based pipeline in cluster_functions.
            # The backend flag is accepted for parity and future extension.
            cfg_pcx = dict(self.config)
            cfg_pcx["out_path"] = os.path.join(self.config["out_path"], self.run_name, "pcx")
            plot_pcx(self.model, cfg_pcx, run=self.wandb_run)
        if self.config.get("diff"):
            plot_triptych_from_merge(
                self.config["data_dir"],
                self.config["patient"],
                self.config["gene"],
                os.path.join(self.config["out_path"], self.run_name, "diff"),
                is_online=bool(self.wandb_run),
                wandb_run=self.wandb_run,
            )

        if self.config.get("scatter"):
            cfg_scatter = dict(self.config)
            cfg_scatter["out_path"] = os.path.join(self.config["out_path"], self.run_name, "scatter")
            plot_scatter(
                cfg_scatter,
                self.model,
                wandb_run=self.wandb_run
            )
        if self.config.get("forward_to_csv"):
            patients = sorted([
                f.name for f in os.scandir(self.config["data_dir"]) if f.is_dir() and not f.name.startswith((".", "_"))
            ])

            genes = self.config["model_config"]["genes"]
            run_name = self.run_name
            image_size = int(self.config["model_config"].get("image_size", 224))

            device = auto_device(self.model)

            for p in patients:
                _ = generate_results(
                    model=self.model,
                    device=device,
                    data_dir=self.config["data_dir"],
                    run_name=run_name,
                    out_path=self.config["out_path"],
                    patient=p,
                    genes=genes,
                    meta_data_dir=self.config["meta_data_dir"],
                    gene_data_filename=self.config["gene_data_filename"],
                    image_size=image_size,
                )
        if self.config.get("lxt"):
            # Delegate to dedicated LXT plotting function
            plot_lxt(self.model, self.config, run=self.wandb_run)
