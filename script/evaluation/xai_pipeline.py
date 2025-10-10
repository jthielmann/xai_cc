import torch

from script.evaluation.crp_plotting import plot_crp
from script.evaluation.lrp_plotting import plot_lrp
from script.evaluation.pcx_plotting import plot_pcx
from script.evaluation.tri_plotting import plot_triptych_from_merge
from script.model.model_factory import get_encoder


class XaiPipeline:
    def __init__(self, config, run):
        self.config = config
        self.wandb_run = run
        self.model = self._load_model()

    def _load_model(self):
        encoder_type = self.config["encoder_type"]
        return get_encoder(encoder_type)

    def run(self):
        if self.config.get("lrp", False):
            plot_lrp(self.model, run=self.wandb_run)
        if self.config.get("crp", False):
            plot_crp(self.model, run=self.wandb_run)
        if self.config.get("pcx", False):
            plot_pcx(self.model, self.config, run=self.wandb_run)
        if self.config.get("diff", False):
            plot_triptych_from_merge(
                self.config["data_dir"],
                self.config["patient"],
                self.config["gene"],
                self.config["out_path"],
                is_online=bool(self.wandb_run),
                wandb_run=self.wandb_run,
            )
