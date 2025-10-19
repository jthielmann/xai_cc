import json
import logging
import os
import sys
import warnings
from contextlib import nullcontext
from io import BytesIO
from typing import Dict, Any, Mapping, Iterable, Optional

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from PIL import Image
from torch.optim.lr_scheduler import OneCycleLR
import pandas as pd

from script.model.lit_ae import SparseAutoencoder
from script.model.lit_model_helpers import (
    append_row_with_schema,
    build_results_row,
    build_scatter_appendix,
    bytes2gb,
    compute_per_gene_pearson,
    estimate_per_sample_activations,
    measure_peak_train_step,
    select_per_gene_values,
)
from script.evaluation.scatter_plotting import make_scatter_figure
from script.model.loss_functions import MultiGeneWeightedMSE, PearsonCorrLoss, WeightedHuberLoss
from script.model.model_factory import get_encoder, infer_encoder_out_dim

import logging
log = logging.getLogger(__name__)

sys.path.insert(0, '..')

def load_lit_regressor(config, state_dicts: Dict[str, Any]):
    model = get_model(config)

    encoder_state = state_dicts.get("encoder")
    if encoder_state is None:
        raise ValueError("Missing 'encoder' state dict for load_lit_regressor")
    if encoder_state and encoder_state.keys() and not next(iter(encoder_state)).startswith("encoder."):
        encoder_state = {f"encoder.{k}": v for k, v in encoder_state.items()}
    model.load_state_dict(encoder_state, strict=False)

    sae_state = state_dicts.get("sae")
    if sae_state and getattr(model, "sae", None):
        model.sae.load_state_dict(sae_state)

    head_state = state_dicts.get("gene_heads")
    if head_state:
        for gene_name, gene_dict in head_state.items():
            head_module = getattr(model, gene_name, None)
            if head_module is None:
                continue
            head_module.load_state_dict(gene_dict)

    return model

# Backward compatibility alias; prefer load_lit_regressor in new code
def load_model(config, state_dicts: Dict[str, Any]):
    return load_lit_regressor(config, state_dicts)

def get_model(config):
    return GeneExpressionRegressor(config)

class GeneExpressionRegressor(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        default_lr = self.config.get("head_lr", 1e-3)
        self.lrs = None
        self.encoder = get_encoder(self.config["encoder_type"])
        self.freeze_encoder = bool(self.config.get('freeze_encoder', False))
        raw_layer_names = self.config.get("encoder_finetune_layer_names", [])
        if isinstance(raw_layer_names, str):
            raw_layer_names = [raw_layer_names]
        self.encoder_finetune_layer_names = [str(n) for n in raw_layer_names]
        self.encoder_finetune_layers = int(self.config.get("encoder_finetune_layers", 0) or 0)
        self.encoder_unfrozen_groups: list[str] = []
        self.encoder_is_fully_frozen: bool = False
        self._apply_encoder_freeze_policy()
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
        loss_switch = str(self.config["loss_fn_switch"]).lower()
        if loss_switch == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_switch in {"wmse", "weighted mse"}:
            # Strict WMSE: expects per-sample weights from the dataset
            wmse_kwargs = {
                "eps": float(self.config.get("wmse_eps", 1e-8)),
                "reduction": str(self.config.get("wmse_reduction", "mean")).lower(),
                "normalize": str(self.config.get("wmse_normalize", "global")).lower(),
                "clip_weights": self.config.get("wmse_clip_weights"),
                "check_finite": bool(self.config.get("wmse_check_finite", True)),
            }
            self.loss_fn = MultiGeneWeightedMSE(**wmse_kwargs)
        elif loss_switch in {"weighted huber", "wmse huber", "weighted smoothl1"}:
            huber_kwargs = {
                "delta": float(self.config.get("weighted_huber_delta", 1.0)),
                "eps": float(self.config.get("wmse_eps", 1e-8)),
                "reduction": str(self.config.get("wmse_reduction", "mean")).lower(),
                "normalize": str(self.config.get("wmse_normalize", "global")).lower(),
                "clip_weights": self.config.get("wmse_clip_weights"),
                "check_finite": bool(self.config.get("wmse_check_finite", True)),
            }
            self.loss_fn = WeightedHuberLoss(**huber_kwargs)
        elif loss_switch == "pearson":
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

    def _apply_encoder_freeze_policy(self) -> None:
        # Build/refresh param groups and their order
        self._encoder_param_groups, self._encoder_param_group_order = self._build_encoder_param_groups()

        # Set all params frozen or unfrozen in one pass
        all_trainable = not self.freeze_encoder
        for p in self.encoder.parameters():
            p.requires_grad = all_trainable

        groups = []

        if self.freeze_encoder:
            # Start fully frozen, then selectively unfreeze
            if self.encoder_finetune_layer_names:
                groups += self._unfreeze_groups(self.encoder_finetune_layer_names)

            if self.encoder_finetune_layers > 0:
                groups += self._unfreeze_last_n_groups(
                    self.encoder_finetune_layers,
                    exclude=set(groups)  # avoid re-unfreezing duplicates
                )

            # Remove duplicates while preserving order
            groups = list(dict.fromkeys(groups))

            if groups:
                log.info("Partially finetuning encoder groups: %s", ", ".join(groups))
            else:
                log.info("Encoder fully frozen (no finetuning groups specified)")
        else:
            # Fully trainable encoder
            groups = list(self._encoder_param_group_order)

        self.encoder_unfrozen_groups = groups

        # Compute freeze status and switch mode accordingly
        self.encoder_is_fully_frozen = not any(p.requires_grad for p in self.encoder.parameters())
        self.encoder.train(not self.encoder_is_fully_frozen)


    def _build_encoder_param_groups(self) -> tuple[dict[str, list[nn.Parameter]], list[str]]:
        groups: dict[str, list[nn.Parameter]] = {}
        order: list[str] = []
        for name, param in self.encoder.named_parameters():
            group = self._parameter_group_from_name(name)
            if group not in groups:
                groups[group] = []
                order.append(group)
            groups[group].append(param)
        return groups, order

    @staticmethod
    def _parameter_group_from_name(name: str) -> str:
        parts = name.split('.')
        if len(parts) > 2 and parts[1] == "layer" and parts[2].isdigit():
            return ".".join(parts[:3])
        if len(parts) > 1 and parts[1].isdigit():
            first = parts[0]
            if first in {"blocks", "layers", "stages", "encoder_layers"} or first.endswith("blocks") or first.endswith("layers"):
                return ".".join(parts[:2])
        return parts[0]

    def _unfreeze_groups(self, target_groups: Iterable[str]) -> list[str]:
        matched: list[str] = []
        available = self._encoder_param_group_order
        for raw_target in target_groups:
            target = str(raw_target)
            if not target:
                continue
            hits = [g for g in available if g == target]
            if not hits:
                hits = [g for g in available if g.endswith(target)]
            if not hits:
                log.warning(
                    "Requested encoder group '%s' not found; available groups include: %s",
                    target,
                    ", ".join(available[-5:])
                )
                continue
            for group in hits:
                if group in matched:
                    continue
                for param in self._encoder_param_groups.get(group, []):
                    param.requires_grad = True
                matched.append(group)
        return matched

    def _unfreeze_last_n_groups(self, count: int, exclude: Optional[Iterable[str]] = None) -> list[str]:
        if count <= 0:
            return []
        exclude_set = set(exclude or [])
        selected: list[str] = []
        for group in reversed(self._encoder_param_group_order):
            if group in exclude_set:
                continue
            selected.append(group)
            if len(selected) >= count:
                break
        selected.reverse()
        for group in selected:
            for param in self._encoder_param_groups.get(group, []):
                param.requires_grad = True
        return selected

    def train(self, mode: bool = True):
        """Ensure frozen encoders do not update BatchNorm running stats.

        When `freeze_encoder` is True, keep the encoder in eval mode even
        while the overall module is in train mode.
        """
        super().train(mode)
        if getattr(self, "encoder_is_fully_frozen", False):
            # keep encoder/BNS in eval to prevent running-stat drift
            try:
                self.encoder.eval()
                for m in self.encoder.modules():
                    if isinstance(m, nn.modules.batchnorm._BatchNorm):
                        m.eval()
            except Exception:
                # best-effort; don't crash training if encoder lacks BN
                pass
        return self


    def configure_optimizers(self):
        groups = []
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        if encoder_params:
            groups.append({
                "params": encoder_params,
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
        self.val_loss_weight_sum = 0.0
        self.y_hats, self.ys = [], []

    def forward(self, x):
        freeze_context = getattr(self, "encoder_is_fully_frozen", False)
        cm = torch.no_grad if freeze_context else nullcontext
        with cm():
            z = self.encoder(x)
        if isinstance(z, (list, tuple)):
            z = z[0]
        if hasattr(z, "last_hidden_state"):
            z = z.last_hidden_state
        if z.ndim == 4:
            z = torch.flatten(z, 1)
        elif z.ndim == 3:
            z = z.mean(dim=1)
        if z.ndim != 2:
            raise RuntimeError(f"Normalized encoder output must be 2D, got {tuple(z.shape)}")
        if self.sae:
            z = self.sae(z)

        outs = [getattr(self, g)(z) for g in self.genes]
        out = torch.cat(outs, dim=1)

        return out

    def training_step(self, batch, batch_idx):
        loss, _, _, stats = self._step(batch)
        self.log('train_' + self.config['loss_fn_switch'], loss, on_step=True, on_epoch=True, prog_bar=True)
        if stats:
            self._log_weight_stats(stats, prefix="train_wmse", on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y, stats = self._step(batch)
        self.log('val_' + self.config['loss_fn_switch'], loss, on_epoch=True)

        bs = y.size(0)
        loss_switch = self.config["loss_fn_switch"].lower()
        if loss_switch in {"wmse", "weighted mse", "weighted huber", "wmse huber", "weighted smoothl1"} and stats:
            if "numerator" in stats and "denominator" in stats:
                self.val_loss_total += float(stats["numerator"])
                self.val_loss_weight_sum += float(stats["denominator"])
            elif "numerator_vector" in stats and "denominator_vector" in stats:
                self.val_loss_total += float(stats["numerator_vector"].sum())
                self.val_loss_weight_sum += float(stats["denominator_vector"].sum())
            else:
                self.val_loss_total += float(loss.detach()) * bs
                self.val_loss_count += bs
        elif loss_switch in {"mse"}:
            self.val_loss_total += float(loss.detach()) * bs
            self.val_loss_count += bs
        else:
            # average over batches
            self.val_loss_total += float(loss.detach())
            self.val_loss_count += 1

        if stats:
            self._log_weight_stats(stats, prefix="val_wmse", on_step=False, on_epoch=True)

        self.y_hats.append(y_hat.detach().float().cpu())
        self.ys.append(y.detach().float().cpu())
        return loss

    def _log_wandb_artifacts(self):
        # Never log artifacts in debug mode
        if bool(self.config.get("debug")):
            return
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

    def _log_weight_stats(self, stats: Optional[dict], *, prefix: str, on_step: bool, on_epoch: bool) -> None:
        if not stats:
            return
        for key in ("weight_mean", "weight_max", "weight_min", "weight_nonzero_frac"):
            value = stats.get(key)
            if value is None:
                continue
            self.log(prefix + "_" + key, float(value), on_step=on_step, on_epoch=on_epoch, prog_bar=False)

    def _step(self, batch):
        loss_switch = str(self.config.get("loss_fn_switch", "")).lower()
        if loss_switch in {"wmse", "weighted mse"}:
            if not (isinstance(batch, (list, tuple)) and len(batch) == 3):
                raise ValueError(
                    "Expected (x, y, w) batch for WMSE. Ensure 'lds_weight_csv' is set and DataModule passes weights."
                )
            x, y, w = batch
        elif loss_switch in {"weighted huber", "wmse huber", "weighted smoothl1"}:
            if not (isinstance(batch, (list, tuple)) and len(batch) in {2, 3}):
                raise ValueError(
                    "Expected (x, y, w) or (x, y) batch for weighted Huber. Provide weights or switch to unweighted loss."
                )
            if len(batch) == 3:
                x, y, w = batch
            else:
                x, y = batch
                w = None
        else:
            # Plain MSE or other losses generally expect (x, y) only, but we tolerate (x, y, w) by ignoring w.
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    x, y = batch
                    w = None
                elif len(batch) == 3:
                    x, y, maybe_w = batch
                    if isinstance(maybe_w, torch.Tensor):
                        warnings.warn(
                            f"Received sample weights for loss '{self.config.get('loss_fn_switch')}', ignoring them.",
                            RuntimeWarning,
                        )
                    w = None
                else:
                    raise ValueError(
                        f"Unsupported batch structure of length {len(batch)} for loss '{self.config.get('loss_fn_switch')}'."
                    )
            else:
                raise ValueError(
                    f"Expected batch to be tuple/list, got type {type(batch).__name__} for loss '{self.config.get('loss_fn_switch')}'."
                )

        y_hat = self(x)

        if y.dim() == 1:
            y = y.unsqueeze(1)

        if y_hat.shape != y.shape:
            raise ValueError(f"Shape mismatch: {y_hat.shape} vs {y.shape}")

        stats = None
        if loss_switch in {"wmse", "weighted mse"}:
            if w is None:
                raise ValueError("Missing sample weights for WMSE.")
            result = self.loss_fn(y_hat, y, w, return_stats=True)
            if isinstance(result, tuple):
                loss, stats = result
            else:
                loss, stats = result, None
        elif loss_switch in {"weighted huber", "wmse huber", "weighted smoothl1"}:
            result = self.loss_fn(y_hat, y, w, return_stats=True)
            if isinstance(result, tuple):
                loss, stats = result
            else:
                loss, stats = result, None
        else:
            loss = self.loss_fn(y_hat, y)
        return loss, y_hat, y, stats

    def _update_best(self, loss_sum: float, epoch: int, out_path: str, r_mean: float, per_gene_r: list[float]) -> bool:
        """Update best checkpoint if improved. Returns True if a new best was set."""
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
            return True
        return False

    def _save_outputs_csv(self, y_pred: torch.Tensor, y_true: torch.Tensor, split: str) -> None:
        """Persist predictions and targets to CSV next to best_model.pth.

        - Overwrites the file on each call (used for val on new best).
        - Columns: pred_{gene}, target_{gene} for each gene in order.
        """
        # Resolve base directory next to best_model.pth
        best_path = self.best_model_path or os.path.join(self.config.get("out_path", "."), "best_model.pth")
        base_dir = os.path.dirname(best_path) if best_path else self.config.get("out_path", ".")
        os.makedirs(base_dir, exist_ok=True)

        # Ensure tensors on CPU and as numpy
        yp = torch.as_tensor(y_pred).detach().cpu().float().numpy()
        yt = torch.as_tensor(y_true).detach().cpu().float().numpy()

        # Build a wide dataframe with named columns
        pred_cols = [f"pred_{g}" for g in self.genes]
        targ_cols = [f"target_{g}" for g in self.genes]
        data = {name: yp[:, i] for i, name in enumerate(pred_cols)}
        data.update({name: yt[:, i] for i, name in enumerate(targ_cols)})
        df = pd.DataFrame(data)

        out_csv = os.path.join(base_dir, f"best_{split}_outputs.csv")
        try:
            df.to_csv(out_csv, index=False)
            log.info("Saved %s outputs next to best model: %s", split, out_csv)
        except Exception as e:
            log.exception("Failed to save %s outputs CSV to %s: %s", split, out_csv, e)

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

        # Concatenate all batches to compute dataset-level metrics
        y_hat = torch.cat(self.y_hats, dim=0).float()
        y_true = torch.cat(self.ys, dim=0).float()

        per_gene_r = compute_per_gene_pearson(y_hat, y_true)
        self.last_r = list(per_gene_r)
        if self.is_online:
            self.log_dict({f"pearson_{g}": r for g, r in zip(self.genes, per_gene_r)}, on_epoch=True)

        loss_switch = str(self.config.get("loss_fn_switch", "")).lower()
        if loss_switch in {"wmse", "weighted mse", "weighted huber", "wmse huber", "weighted smoothl1"}:
            denom_base = getattr(self.loss_fn, "eps", 1e-8)
            denominator = max(self.val_loss_weight_sum, denom_base)
            val_loss_mean = float(self.val_loss_total / denominator) if denominator > 0 else float("nan")
        elif loss_switch == "mse":
            denominator = max(self.val_loss_count, 1)
            val_loss_mean = float(self.val_loss_total / denominator)
        else:
            denominator = max(self.val_loss_count, 1)
            val_loss_mean = float(self.val_loss_total / denominator)
        r_mean = float(np.nanmean(per_gene_r)) if per_gene_r else float("nan")
        criterion = -r_mean if "pearson" in loss_switch else val_loss_mean

        # Update "best" (expects lower-is-better) and save CSV for this best epoch
        is_new_best = self._update_best(criterion, int(self.current_epoch), out_path, r_mean, per_gene_r)
        if is_new_best:
            self._save_outputs_csv(y_hat, y_true, split="val")

        # Track best Pearson mean across epochs independently of loss type
        if np.isfinite(r_mean):
            if not hasattr(self, "best_pearson_mean"):
                self.best_pearson_mean = float("-inf")
                self.best_pearson_epoch = None
                self.best_pearson_per_gene = [float("nan")] * len(self.genes)
            if r_mean > (self.best_pearson_mean if np.isfinite(getattr(self, "best_pearson_mean", float("nan"))) else float("-inf")):
                self.best_pearson_mean = float(r_mean)
                self.best_pearson_epoch = int(self.current_epoch)
                self.best_pearson_per_gene = [float(x) for x in per_gene_r]
                if self.is_online and wandb.run:
                    wandb.run.summary.update({
                        "best_pearson_mean": self.best_pearson_mean,
                        "best_pearson_epoch": self.best_pearson_epoch,
                    })
                    for g, r in zip(self.genes, self.best_pearson_per_gene):
                        wandb.run.summary[f"best_pearson_{g}"] = float(r)

        # Log current epoch Pearson mean for visibility
        self.log("val_pearson_mean", r_mean, on_epoch=True)

        # Optional scatter plots
        if self.config.get("generate_scatters", False):
            sweep_cfg = getattr(getattr(self, "wandb_run", None), "config", {})
            appendix = build_scatter_appendix(self.config, sweep_cfg.get("sweep_parameter_names", []))

            for gi, gene in enumerate(self.genes):
                yi, ti = y_hat[:, gi], y_true[:, gi]
                r_val = float(per_gene_r[gi])
                loss_g = float(torch.mean((yi - ti) ** 2))
                title_parts = [f"ep {self.current_epoch}", gene]
                if appendix:
                    title_parts.append(appendix)
                fig = make_scatter_figure(
                    yi.cpu().numpy(),
                    ti.cpu().numpy(),
                    loss_g,
                    r_val,
                    " — ".join(title_parts),
                )
                self._log_scatter_to_wandb(fig, gene, int(self.current_epoch))

        # Reset accumulators
        self.y_hats.clear()
        self.ys.clear()

    # ---------------------- Test hooks (optional) ----------------------
    def on_test_epoch_start(self):
        self.test_y_hats, self.test_ys = [], []

    def test_step(self, batch, batch_idx):
        # Reuse the common _step path for consistency and log test loss per-batch
        loss, y_hat, y, stats = self._step(batch)
        self.log('test_' + self.config['loss_fn_switch'], loss, on_epoch=True)
        self.test_y_hats.append(y_hat.detach().float().cpu())
        self.test_ys.append(y.detach().float().cpu())
        return loss

    def on_test_epoch_end(self):
        if not self.test_y_hats or not self.test_ys:
            return
        y_hat = torch.cat(self.test_y_hats, dim=0).float()
        y_true = torch.cat(self.test_ys, dim=0).float()
        # Always write test outputs if test is run; does not imply best update here
        self._save_outputs_csv(y_hat, y_true, split="test")
        self.test_y_hats.clear()
        self.test_ys.clear()

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
        is_debug = bool(self.config.get("debug"))
        if self.is_online and hasattr(self, "table") and not is_debug:
            wandb.log({"scatter_table": self.table})
        results_root = "../results"
        if is_debug:
            csv_paths = [os.path.join(results_root, "debug.csv")]
        else:
            project = self.config.get("project", "project")
            csv_paths = [
                os.path.join(results_root, "all.csv"),
                os.path.join(results_root, project, "results.csv"),
            ]

        best_pearson_mean = (
            float(self.best_pearson_mean)
            if hasattr(self, "best_pearson_mean") and np.isfinite(getattr(self, "best_pearson_mean", float("nan")))
            else float(self.best_r_mean)
        )

        metrics = {
            "best_epoch": int(self.best_epoch) if getattr(self, "best_epoch", None) is not None else int(self.current_epoch),
            "best_loss": float(self.best_loss),
            "pearson_mean": best_pearson_mean,
            "best_model_path": self.best_model_path or os.path.join(self.config["out_path"], "best_model.pth"),
            "wandb_url": wandb.run.url if self.is_online and wandb.run else "",
            "best_pearson_epoch": int(self.best_pearson_epoch) if getattr(self, "best_pearson_epoch", None) is not None else None,
        }

        per_gene_values = select_per_gene_values(
            getattr(self, "best_pearson_per_gene", None),
            getattr(self, "best_r", None),
            getattr(self, "last_r", None),
            len(self.genes),
        )

        tuned_lrs = self.lrs if isinstance(getattr(self, "lrs", None), dict) else None
        row = build_results_row(self.config, self.genes, metrics, per_gene_values, tuned_lrs)

        for path in csv_paths:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                append_row_with_schema(path, row)
                logging.info("logged results into %s", path)
            except Exception as e:
                logging.exception("failed to log results into %s: %s", path, e)

        if not is_debug:
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
