import os
from typing import Any, Dict, Optional

#from script.evaluation.crp_plotting import plot_crp, plot_crp_zennit, plot_crp2
from script.evaluation.lrp_plotting import plot_lrp, plot_lrp_custom
from script.evaluation.lxt_plotting import plot_lxt
from script.evaluation.pcx_plotting import plot_pcx
from script.evaluation.tri_plotting import plot_triptych_from_merge
from script.evaluation.scatter_plotting import plot_scatter
from script.evaluation.generate_results import generate_results, log_patient_hist_from_csv
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
        if self.config.get("forward_to_csv"):
            patients = [f.name for f in os.scandir(self.config["data_dir"]) if f.is_dir() and not str(f).startswith(".") and not str(f).startswith("_")]

            results_dir = self.config.get(
                "results_dir",
                os.path.join(self.config.get("out_path", "."), "forward_results")
            )
            genes = self.config["model_config"]["genes"]

            device = auto_device(self.model)

            for p in patients:
                csv_path = generate_results(
                    model=self.model,
                    device=device,
                    data_dir=self.config["data_dir"],
                    run_name=self.config["model_config"]["name"],
                    patient=p,
                    genes=genes,
                    meta_data_dir=self.config["meta_data_dir"],
                    gene_data_filename=self.config["gene_data_filename"]
                )
                """
                # Log prediction vs. original histograms to W&B (and save an image)
                log_patient_hist_from_csv(
                    results_csv=csv_path,
                    data_dir=self.config["data_dir"],
                    patient=p,
                    genes=genes,
                    results_dir=results_dir
                )
                """
        if self.config.get("lxt"):
            # Delegate to dedicated LXT plotting function
            plot_lxt(self.model, self.config, run=self.wandb_run)
