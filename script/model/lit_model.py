import numpy as np
import torch
import lightning as L
import torch.nn as nn
import torchvision
import torchmetrics
import torch.optim as optim
import wandb
from script.data_processing.image_transforms import get_transforms
from script.model.model_factory import get_encoder, infer_encoder_out_dim
import os
from script.data_processing.process_csv import generate_results_patient_from_loader
from script.train.generate_plots import generate_hists_2
import matplotlib.pyplot as plt
from script.data_processing.data_loader import get_dataset, load_best_smoothing, load_gene_weights
from torch.utils.data import DataLoader
import random
import sys
from io import BytesIO
from PIL import Image
from torch.optim.lr_scheduler import OneCycleLR, ChainedScheduler
sys.path.insert(0, '..')
from script.model.loss_functions import MultiGeneWeightedMSE, PearsonCorrLoss
from script.model.lit_ae import SparseAutoencoder
from typing import Dict, Any
from torchmetrics.functional import pearson_corrcoef
import scipy

def load_model(path: str, config: Dict[str, Any]) -> L.LightningModule:
    device = "cpu"
    model = GeneExpressionRegressor(config)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model

def load_model2(config: Dict[str, Any], model_name = "/best_model.pth") -> L.LightningModule:
    device = "cpu"
    model = GeneExpressionRegressor(config)
    state = torch.load(config["out_path"] + model_name, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model

def get_model(config):
    return GeneExpressionRegressor(config)

class GeneExpressionRegressor(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        default_lr = self.config.get("head_lr", 1e-3)
        self.lrs = None
        self.encoder = get_encoder(self.config["encoder_type"])
        self.freeze_encoder = self.config['freeze_encoder']
        if self.freeze_encoder:
            for p in self.encoder.parameters(): p.requires_grad = False
        out_dim_encoder = infer_encoder_out_dim(self.encoder)
        for gene in self.config['genes']:
            relu_type = nn.LeakyReLU if self.config.get('use_leaky_relu') else nn.ReLU
            relu_instance = relu_type()

            layer = (
                nn.Sequential(relu_instance, nn.Linear(out_dim_encoder, 1))
                if self.config['one_linear_out_layer']
                else nn.Sequential(
                    nn.Linear(out_dim_encoder, self.config['middle_layer_features']),
                    relu_instance,
                    nn.Linear(self.config['middle_layer_features'], 1),
                )
            )

            setattr(self, gene, layer)
        if self.config.get("sae", False):
            self.config["d_in"] = out_dim_encoder
            self.sae  = SparseAutoencoder(self.config)
        else:
            self.sae = None


        self.genes = self.config['genes']

        for g in self.genes:
            setattr(self, f"{g}_lr", default_lr)
        if self.config["loss_fn_switch"] == "MSE":
            self.loss_fn = nn.MSELoss()
        elif self.config["loss_fn_switch"] == "WMSE" or self.config["loss_fn_switch"] == "weighted MSE":
            weight_dir = self.config["lds_weight_csv"]
            weights = load_gene_weights(weight_dir, self.genes)
            self.loss_fn = MultiGeneWeightedMSE(weights)
        elif self.config["loss_fn_switch"] == "pearson":
            self.loss_fn = PearsonCorrLoss()
        else:
            raise ValueError(f"loss_fn_switch {self.config['loss_fn_switch']} not implemented")

        self.num_training_batches = 0
        self.current_loss = torch.tensor(0.).to(self.device)
        self.best_loss = float("inf")
        self.is_online = self.config.get('log_to_wandb')

        if self.config.get('generate_scatters', False):
            self.table = wandb.Table(columns=["epoch","gene","lr","bins","scatter_plot"])
        if self.is_online:
            wandb.watch(self, log=None)
        self.encoder_lr = self.config.get("encoder_lr", 1e-3)  # encoder
        default_head_lr = self.config.get("head_lr", 1e-3)  # heads
        for g in self.config["genes"]:
            setattr(self, f"{g}_lr", default_head_lr)


    def configure_optimizers(self):
        groups = []
        if not self.freeze_encoder:
            groups.append({
                "params": self.encoder.parameters(),
                "lr": self.encoder_lr
            })
        for g in self.genes:
            groups.append({
                "params": getattr(self, g).parameters(),
                "lr": getattr(self, f"{g}_lr")
            })
        opt = torch.optim.AdamW(groups)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=[pg["lr"] for pg in groups],
            total_steps=self.trainer.estimated_stepping_batches
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def on_fit_start(self):          # ← runs before the first batch
        os.makedirs(self.config["out_path"], exist_ok=True)

    def on_validation_epoch_start(self):
        self.current_loss = 0.0
        self.y_hats = []
        self.ys = []

    def forward(self, x):
        z = self.encoder(x)
        if self.sae:
            z = self.sae(z)

        outs = [getattr(self, g)(z) for g in self.genes]  # each (B, 1)
        out = torch.cat(outs, dim=1)

        return out

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch)
        self.log('train_' + self.config['loss_fn_switch'], loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._step(batch)
        self.log('val_' + self.config['loss_fn_switch'], loss, on_epoch=True)

        self.y_hats.append(y_hat.detach().float().cpu())
        self.ys.append(y.detach().float().cpu())
        self.current_loss += loss.detach().item()
        return loss

    def _step(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        y_hat = self(x)

        if y.dim() == 1:
            y = y.unsqueeze(1)

        if y_hat.shape != y.shape:
            raise ValueError(f"Shape mismatch: {y_hat.shape} vs {y.shape}")

        loss = self.loss_fn(y_hat, y)
        return loss, y_hat, y

    def on_validation_epoch_end(self):
        # Skip sanity check epoch
        if not hasattr(self, "sanity_skipped"):
            self.sanity_skipped = True
            return

        torch.save(self.state_dict(), os.path.join(self.config["out_path"], "latest.pth"))
        if self.current_loss < self.best_loss:
            self.best_loss = self.current_loss
            best_model_path = os.path.join(self.config["out_path"], "best_model.pth")
            torch.save(self.state_dict(), best_model_path)
            if self.is_online:
                wandb.run.summary["best_val_loss"] = self.best_loss
                wandb.run.summary["best_val_epoch"] = self.current_epoch
                wandb.log({"epoch": self.current_epoch})

        y_hat = torch.cat(self.y_hats, dim=0).float()
        y_true = torch.cat(self.ys, dim=0).float()

        with torch.no_grad():
            # Per-gene Pearson once
            per_gene_r = []
            for i in range(y_hat.shape[1]):
                yi, ti = y_hat[:, i], y_true[:, i]
                if yi.std(unbiased=False) == 0 or ti.std(unbiased=False) == 0:
                    per_gene_r.append(float("nan"))
                else:
                    per_gene_r.append(float(pearson_corrcoef(yi, ti)))

            if self.is_online:
                self.log_dict({f"pearson_{g}": float(r) for g, r in zip(self.genes, per_gene_r)}, on_epoch=True)

            if self.config.get("generate_scatters", False):
                ABBR = {
                    "learning_rate": "lr", "batch_size": "bs", "dropout_rate": "dr",
                    "loss_fn_swtich": "loss", "encoder_type": "encdr",
                    "middle_layer_features": "mfeatures", "gene_data_filename": "file",
                    "freeze_encoder": "f_encdr", "one_linear_out_layer": "1linLr",
                    "use_leaky_relu": "lkReLu", "use_early_stopping": "eStop",
                }
                sweep_keys = getattr(getattr(self, "wandb_run", None), "config", {}).get("sweep_parameter_names", [])
                appendix = " | ".join(
                    f"{ABBR.get(k, k)}={self.config.get(k)}" for k in sweep_keys) if sweep_keys else ""
                table = getattr(self, "table", None)

                for gi, gene in enumerate(self.genes):
                    yi, ti = y_hat[:, gi], y_true[:, gi]
                    r_value = float(per_gene_r[gi])
                    loss = float(self.loss_fn(yi, ti))

                    yh = yi.detach().cpu().numpy()
                    yt = ti.detach().cpu().numpy()
                    lo = float(min(yh.min(), yt.min()))
                    hi = float(max(yh.max(), yt.max()))
                    if lo == hi:
                        lo, hi = lo - 1.0, hi + 1.0

                    fig, ax = plt.subplots()
                    ax.scatter(yh, yt, s=8)
                    ax.plot([lo, hi], [lo, hi], linewidth=1)
                    ax.text(0.02, 0.98, f"loss: {loss:.3f}\nr: {r_value:.3f}",
                            transform=ax.transAxes, va="top", ha="left", fontsize="small")
                    title = " — ".join([f"ep {self.current_epoch}", gene] + ([appendix] if appendix else []))
                    ax.set_title(title)
                    ax.set_xlabel("output")
                    ax.set_ylabel("target")
                    ax.set_aspect("equal", adjustable="box")

                    from io import BytesIO
                    with BytesIO() as buf:
                        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                        plt.close(fig)
                        buf.seek(0)
                        if self.is_online and table is not None:
                            img = Image.open(buf)
                            table.add_data(
                                self.current_epoch,
                                gene,
                                self.config.get("learning_rate"),
                                self.config.get("bins", 0),
                                wandb.Image(img, caption=gene),
                            )
                            img.close()

        # prevent accumulation across epochs
        self.y_hats.clear()
        self.ys.clear()

        # prevent accumulation across epochs
        self.y_hats.clear()
        self.ys.clear()

    def on_train_epoch_end(self):
        lrs = [g["lr"] for g in self.trainer.optimizers[0].param_groups]
        for idx, lr in enumerate(lrs):
            # Logs: lr_group_0, lr_group_1, …
            self.log(f"lr_group_{idx}", lr, on_epoch=True, prog_bar=False)

        for g in self.genes:
            w_norm = getattr(self, g)[-1].weight.norm()
            self.log(f"{g}_w_norm", w_norm, on_epoch=True, prog_bar=False)

        if not self.config.get('debug'):
            torch.save(self.state_dict(), self.config["out_path"] + "/latest.pth")

    def on_train_end(self):
        if self.is_online:
            if hasattr(self, 'table'):
                wandb.log({'scatter_table': self.table})


    # to update after lr tuning
    def update_lr(self, lrs):
        self.lrs = lrs
        if not self.freeze_encoder:
            self.encoder_lr = lrs["encoder"]  # encoder
        for g in self.config["genes"]:
            setattr(self, f"{g}_lr", lrs[g])

    def get_lr(self, param_name):
        return self.lrs[param_name]
