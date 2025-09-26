import json
import logging

import numpy as np
import lightning as L
import torch.nn as nn
import wandb
from script.model.model_factory import get_encoder, infer_encoder_out_dim
import os
import matplotlib.pyplot as plt
from script.data_processing.data_loader import load_gene_weights
import sys
from io import BytesIO
from PIL import Image
from torch.optim.lr_scheduler import OneCycleLR
sys.path.insert(0, '..')
from script.model.loss_functions import MultiGeneWeightedMSE, PearsonCorrLoss
from script.model.lit_ae import SparseAutoencoder
from typing import Dict, Any
from torchmetrics.functional import pearson_corrcoef
import pandas as pd

import torch
from contextlib import nullcontext

def bytes2gb(x): return x / (1024**3)

@torch.no_grad()
def estimate_per_sample_activations(model, sample):
    """CPU/any-device estimate of forward outputs that require grad (≈ activations kept for backward)."""
    sizes = []
    handles = []

    def hook(_m, _inp, out):
        def num_bytes(t):
            if not torch.is_tensor(t): return 0
            # assume AMP/bfloat16 for training activations → 2 bytes/elem; change to 4 for fp32
            return t.numel() * 2 if t.requires_grad else 0
        if isinstance(out, (list, tuple)):
            sizes.append(sum(num_bytes(t) for t in out))
        else:
            sizes.append(num_bytes(out))

    for m in model.modules():
        if len(list(m.children())) == 0:
            handles.append(m.register_forward_hook(hook))

    model.eval()
    _ = model(sample)  # forward only; we just need shapes
    for h in handles: h.remove()
    # sum once, then divide by batch to get per-sample
    total_bytes = sum(sizes)
    return total_bytes / sample.shape[0]

def measure_peak_train_step(model, batch, criterion, optimizer, amp=True, device="cuda"):
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device).train()
    x, y = batch
    x, y = x.to(device), y.to(device)

    ctx = torch.cuda.amp.autocast if amp else nullcontext
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    optimizer.zero_grad(set_to_none=True)
    with ctx():
        out = model(x)
        loss = criterion(out, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return torch.cuda.max_memory_allocated(device)  # bytes

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

        self.genes = list(self.config['genes'])
        if len(self.genes) != len(set(self.genes)):
            raise RuntimeError("Duplicate genes in config['genes']")
        self.gene_to_idx = {g: i for i, g in enumerate(self.genes)}
        if not self.gene_to_idx == {g: i for i, g in enumerate(self.genes)}:
            raise RuntimeError("Gene/index mapping drifted!")

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

        self.best_epoch = None
        self.best_r_mean = float("nan")
        self.best_model_path = None

        if self.config.get('generate_scatters', False) and self.is_online:
            self.table = wandb.Table(columns=["epoch", "gene", "lr", "bins", "scatter_plot"])
        if self.is_online:
            wandb.watch(self, log=None)
        self.encoder_lr = self.config.get("encoder_lr", 1e-3)  # encoder
        default_head_lr = self.config.get("head_lr", 1e-3)  # heads
        for g in self.config["genes"]:
            setattr(self, f"{g}_lr", default_head_lr)
        self.best_r: list[float] = [float("nan")] * len(self.genes)
        self.last_r: list[float] = [float("nan")] * len(self.genes)


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
        sched = OneCycleLR(
            opt,
            max_lr=[pg["lr"] for pg in groups],
            total_steps=self.trainer.estimated_stepping_batches
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def on_fit_start(self):
        os.makedirs(self.config["out_path"], exist_ok=True)
        with open(os.path.join(self.config["out_path"], "config.json"), "w") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def on_validation_epoch_start(self):
        self.val_loss_total = 0.0
        self.val_loss_count = 0
        self.y_hats, self.ys = [], []

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

        bs = y.size(0)
        if self.config["loss_fn_switch"].lower() in {"mse", "wmse", "weighted mse"}:
            self.val_loss_total += float(loss.detach()) * bs
            self.val_loss_count += bs
        else:
            # average over batches
            self.val_loss_total += float(loss.detach())
            self.val_loss_count += 1

        self.y_hats.append(y_hat.detach().float().cpu())
        self.ys.append(y.detach().float().cpu())
        return loss

    def _log_wandb_artifacts(self):
        if not (self.is_online and wandb.run):
            return
        run = wandb.run
        cfg_path = os.path.join(self.config["out_path"], "config.json")
        best_path = self.best_model_path or os.path.join(self.config["out_path"], "best_model.pth")
        if not os.path.exists(best_path):
            # ensure we have a file (fallback to current state)
            torch.save(self.state_dict(), best_path)

        art = wandb.Artifact(
            name=f"model-{run.id}",
            type="model",
            metadata={
                "encoder_type": self.config.get("encoder_type"),
                "n_genes": len(self.genes),
                "genes": self.genes,
                "freeze_encoder": self.config.get("freeze_encoder", False),
                "loss_fn": self.config.get("loss_fn_switch"),
                "best_epoch": self.best_epoch,
                "best_val_metric": self.best_loss,
            },
        )
        art.add_file(best_path, name="best_model.pth")
        if os.path.exists(cfg_path):
            art.add_file(cfg_path, name="config.json")

        run.log_artifact(art, aliases=["best", "latest"])

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

    def _compute_per_gene_pearson(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> list[float]:
        y_hat = y_hat.float()
        y_true = y_true.float()
        r = []
        with torch.no_grad():
            for i in range(y_hat.shape[1]):
                yi, ti = y_hat[:, i], y_true[:, i]
                if yi.std(unbiased=False) == 0 or ti.std(unbiased=False) == 0:
                    r.append(float("nan"))
                else:
                    r.append(float(pearson_corrcoef(yi, ti)))
        return r

    def _update_best(self, loss_sum: float, epoch: int, out_path: str, r_mean: float, per_gene_r: list[float]) -> None:
        if loss_sum < getattr(self, "best_loss", float("inf")):
            self.best_loss = float(loss_sum)
            self.best_epoch = int(epoch)
            self.best_r_mean = float(r_mean)
            self.best_r = [float(x) for x in per_gene_r]
            self.best_model_path = os.path.join(out_path, "best_model.pth")
            torch.save(self.state_dict(), self.best_model_path)
            if self.is_online and wandb.run:
                wandb.run.summary.update({"best_val_loss": self.best_loss, "best_val_epoch": self.best_epoch})
                wandb.log({"epoch": self.best_epoch})

    def _scatter_fig(self, yh: np.ndarray, yt: np.ndarray, loss: float, r: float, title: str):
        lo, hi = float(min(yh.min(), yt.min())), float(max(yh.max(), yt.max()))
        if lo == hi: lo, hi = lo - 1.0, hi + 1.0
        fig, ax = plt.subplots()
        ax.scatter(yh, yt, s=8)
        ax.plot([lo, hi], [lo, hi], linewidth=1)
        ax.text(0.02, 0.98, f"loss: {loss:.3f}\nr: {r:.3f}", transform=ax.transAxes,
                va="top", ha="left", fontsize="small")
        ax.set(title=title, xlabel="output", ylabel="target")
        ax.set_aspect("equal", adjustable="box")
        return fig

    def _log_scatter_to_wandb(self, fig, gene: str, epoch: int) -> None:
        if not (self.is_online and getattr(self, "table", None) is not None):
            plt.close(fig)
            return
        with BytesIO() as buf:
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf)
            self.table.add_data(
                epoch,
                gene,
                self.config.get("learning_rate"),
                self.config.get("bins", 0),
                wandb.Image(img, caption=gene),
            ) #["epoch", "gene", "lr", "bins", "scatter_plot"]
            img.close()

    def on_validation_epoch_end(self):
        # Skip the Lightning sanity check run
        if not hasattr(self, "sanity_skipped"):
            self.sanity_skipped = True
            return

        out_path = self.config["out_path"]
        torch.save(self.state_dict(), os.path.join(out_path, "latest.pth"))

        if not self.y_hats or not self.ys:
            raise RuntimeError("no ys")

        with torch.no_grad():
            # Concatenate all batches to compute dataset-level metrics
            y_hat = torch.cat(self.y_hats, dim=0).float()
            y_true = torch.cat(self.ys, dim=0).float()

            per_gene_r = self._compute_per_gene_pearson(y_hat, y_true)
            self.last_r = list(per_gene_r)
            if self.is_online:
                self.log_dict({f"pearson_{g}": r for g, r in zip(self.genes, per_gene_r)}, on_epoch=True)

            try:
                val_loss_mean = float(self.loss_fn(y_hat, y_true))
            except Exception:
                val_loss_mean = float(torch.mean((y_hat - y_true) ** 2))

            r_mean = float(np.nanmean(per_gene_r)) if per_gene_r else float("nan")
            loss_switch = str(self.config.get("loss_fn_switch", "")).lower()
            criterion = -r_mean if "pearson" in loss_switch else val_loss_mean

            # Update "best" (expects lower-is-better)
            self._update_best(criterion, int(self.current_epoch), out_path, r_mean, per_gene_r)

            # Optional scatter plots
            if self.config.get("generate_scatters", False):
                ABBR = {
                    "learning_rate": "lr", "batch_size": "bs", "dropout_rate": "dr",
                    "loss_fn_switch": "loss", "loss_fn_switch": "loss",  # catch both spellings
                    "encoder_type": "encdr", "middle_layer_features": "mfeatures",
                    "gene_data_filename": "file", "freeze_encoder": "f_encdr",
                    "one_linear_out_layer": "1linLr", "use_leaky_relu": "lkReLu",
                    "use_early_stopping": "eStop",
                }
                sweep = getattr(getattr(self, "wandb_run", None), "config", {}).get("sweep_parameter_names", [])
                appendix = " | ".join(f"{ABBR.get(k, k)}={self.config.get(k)}" for k in sweep) if sweep else ""

                for gi, gene in enumerate(self.genes):
                    yi, ti = y_hat[:, gi], y_true[:, gi]
                    r_val = float(per_gene_r[gi])
                    # Use per-gene MSE for the plot to be robust across loss types
                    loss_g = float(torch.mean((yi - ti) ** 2))
                    fig = self._scatter_fig(
                        yi.cpu().numpy(), ti.cpu().numpy(), loss_g, r_val,
                        " — ".join([f"ep {self.current_epoch}", gene] + ([appendix] if appendix else []))
                    )
                    self._log_scatter_to_wandb(fig, gene, int(self.current_epoch))

        # Reset accumulators
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
        # --- W&B scatter table (unchanged) ---
        if self.is_online and hasattr(self, "table"):
            wandb.log({"scatter_table": self.table})
        if self.config.get("debug"):
            return

        csv_path = os.path.join(self.config["model_dir"], "results.csv")

        row = {
            "best_epoch": int(self.best_epoch) if getattr(self, "best_epoch", None) is not None else int(
                self.current_epoch),
            "val_score": float(self.best_loss),
            "pearson_mean": float(self.best_r_mean),
            "out_path": self.config["out_path"],
            "model_path": self.best_model_path or os.path.join(self.config["out_path"], "best_model.pth"),
            "wandb_url": (wandb.run.url if self.is_online and wandb.run else ""),
        }

        # Build per-gene columns robustly
        per_gene_for_row = getattr(self, "best_r", None) or getattr(self, "last_r", None) or []
        if len(per_gene_for_row) != len(self.genes):
            raise RuntimeError(
                f"genes ({len(self.genes)}) vs r ({len(per_gene_for_row)}) length mismatch"
            )

        # Handle duplicate gene names deterministically: pearson_<gene>, pearson_<gene>__1, __2, ...
        seen = {}
        for g, r in zip(self.genes, per_gene_for_row):
            base_key = f"pearson_{g}"
            if g in seen:
                seen[g] += 1
                key = f"{base_key}__{seen[g]}"
            else:
                seen[g] = 0
                key = base_key
            row[key] = float(r)

        # Keep selected hyperparams/metadata
        keep = [
            "dataset",
            "gene_data_filename",
            "encoder_type",
            "freeze_encoder",
            "learning_rate",
            "batch_size",
            "bins",
            "loss_fn_switch",
            "genes",
        ]
        for k in keep:
            if k in self.config:
                row[k] = self.config[k]
        row["hp_json"] = json.dumps(self.config, ensure_ascii=False)

        # --- Append row with schema union (handles changing pearson_* columns) ---
        def _append_row_any_schema(csv_path: str, row_dict: dict):
            df_new = pd.DataFrame([row_dict])
            if os.path.exists(csv_path):
                df_old = pd.read_csv(csv_path)
                # Union of columns, preserving existing order; new columns appended at the end
                all_cols = list(dict.fromkeys(list(df_old.columns) + list(df_new.columns)))
                df_old = df_old.reindex(columns=all_cols)
                df_new = df_new.reindex(columns=all_cols)
                df = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df = df_new
            df.to_csv(csv_path, index=False)

        _append_row_any_schema(csv_path, row)
        logging.info("logged results into %s", csv_path)

        # --- (unchanged) log artifacts ---
        self._log_wandb_artifacts()

    # to update after lr tuning
    def update_lr(self, lrs):
        self.lrs = lrs
        if not self.freeze_encoder:
            self.encoder_lr = lrs["encoder"]  # encoder
        for g in self.config["genes"]:
            setattr(self, f"{g}_lr", lrs[g])

    def get_lr(self, param_name):
        return self.lrs[param_name]
