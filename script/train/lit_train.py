# lit_training.py: Defines the core training pipeline and utilities for
# setting up, running, and summarizing model training with Lightning and W&B.
import gc
# Standard library imports
import os
import json
import logging
from pathlib import Path
import copy

# Third-party imports
import torch
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.callbacks import LearningRateMonitor
from matplotlib import pyplot as plt
from wandb.wandb_run import Run
from typing import Optional, cast, Any
import re
import numpy as np
import torch.nn as nn
from typing import Dict, List
import itertools

# Local application imports
from script.model.lit_model import get_model
from script.data_processing.lit_STDataModule import get_data_module
from script.model.model_factory import get_encoder  # for encoder factory
from script.data_processing.data_loader import get_spatial_dataset
# Prepare a module logger (configuration should be done in entry-point)
log = logging.getLogger(__name__)


def _determine_device() -> str:
    """
    Check available hardware and return 'gpu' or 'mps'.
    Abort if no supported accelerator is available.
    """
    if torch.cuda.is_available():
        return "gpu"
    if torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError("No GPU or MPS device available. Aborting training.")


# --- helpers ---------------------------------------------------------------
def _nanrange(a: np.ndarray) -> tuple[float, float]:
    if a.size == 0:
        return 0.0, 1.0
    return float(np.nanmin(a)), float(np.nanmax(a))

def _ranges(y_label, y_pred, y_diff):
    vmin_lbl, vmax_lbl   = _nanrange(y_label)
    vmin_pred, vmax_pred = _nanrange(y_pred)
    vmax_abs_diff = float(np.nanmax(np.abs(y_diff))) if y_diff.size else 1.0
    return (vmin_lbl, vmax_lbl), (vmin_pred, vmax_pred), (-vmax_abs_diff, vmax_abs_diff)

def _scatter(ax, x, y, vals, vmin, vmax, title, cbar_label):
    sc = ax.scatter(x, y, c=vals, s=8, marker="s", edgecolors="none", vmin=vmin, vmax=vmax)
    ax.set(title=title, xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="box")
    cb = ax.figure.colorbar(sc, ax=ax)
    cb.set_label(cbar_label)

def _single_panel_figure(x, y, vals, vmin, vmax, title, cbar_label):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    _scatter(ax, x, y, vals, vmin, vmax, title, cbar_label)
    return fig

def _plot_triptych_and_log(x, y, y_label, y_pred, patient, gene, out_path, is_online=False, wandb_run=None):
    y_diff = y_pred - y_label
    (vmin_lbl, vmax_lbl), (vmin_pred, vmax_pred), (vmin_diff, vmax_diff) = _ranges(y_label, y_pred, y_diff)

    panels = [
        ("label",      y_label, (vmin_lbl,  vmax_lbl),  "Label"),
        ("prediction", y_pred,  (vmin_pred, vmax_pred), "Prediction"),
        ("diff",       y_diff,  (vmin_diff, vmax_diff), "Diff"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, (name, vals, (vmin, vmax), cbar) in zip(axes, panels):
        _scatter(ax, x, y, vals, vmin, vmax, f"{patient} • {gene} ({name})", cbar)

    out_file = os.path.join(out_path, f"{patient}_{gene}_spatial.png")
    fig.savefig(out_file, dpi=200)

    if is_online and wandb_run is not None:
        singles = {
            name: _single_panel_figure(x, y, vals, vmin, vmax, f"{patient} • {gene} ({name})", cbar)
            for name, vals, (vmin, vmax), cbar in panels
        }
        wandb_run.log({
            f"spatial/{gene}/{patient}/label":      wandb.Image(singles["label"]),
            f"spatial/{gene}/{patient}/prediction": wandb.Image(singles["prediction"]),
            f"spatial/{gene}/{patient}/diff":       wandb.Image(singles["diff"]),
            f"spatial/{gene}/{patient}/triptych":   wandb.Image(fig),
        })
        for f in singles.values():
            plt.close(f)

    plt.close(fig)
    log.info("Saved spatial plot: %s", out_file)
# ---------------------------------------------------------------------------


class TrainerPipeline:
    def __init__(self, config: dict, run: wandb.sdk.wandb_run.Run):

        # early exit
        required = [
            "train_samples", "val_samples",
            "data_dir", "batch_size", "epochs",
            "loss_fn_switch", "encoder_type"
        ]
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"Missing config keys: {missing}")

        self.wandb_run = run
        self.config    = config
        self.debug     = self.config.get("debug")

        self.is_sweep = hasattr(self.wandb_run.config, "sweep_parameter_names")

        self.is_online = self.config.get("log_to_wandb")
        if not self.is_online:
            self.project = "local_run"
            self.logger = None
        else: # online logging to wandb
            self.project   = self.wandb_run.project
            if self.is_sweep:
                abbr = {
                    "learning_rate": "lr",
                    "batch_size": "bs",
                    "dropout_rate": "dr",
                    "loss_fn_swtich": "loss",
                    "encoder_type": "encdr",
                    "middle_layer_features": "mfeatures",
                    "gene_data_filename": "file",
                    "freeze_encoder": "f_encdr",
                    "one_linear_out_layer": "1linLr",
                    "use_leaky_relu": "lkReLu",
                    "use_early_stopping": "eStop"
                }

                parts = []
                for k in self.wandb_run.config.sweep_parameter_names:
                    short = abbr.get(k, k)
                    val = self.config.get(k)
                    parts.append(f"{short}={val}")

                name = ", ".join(parts)
            else:
                name = self.config["project"] + " " + self.config["name"]
            self.wandb_run.name = name
            self.wandb_run.notes = f"Training {self.wandb_run.name} on {self.wandb_run.project}"
            self.logger = WandbLogger(
                project=self.wandb_run.project,
                name=name
            )



        self.device  = _determine_device()
        self.out_path = self._prepare_output_dir()

    def _run_name_to_dir(self, run_name: str, base_dir: str = "outputs") -> str:
        # Split into key=value parts and strip spaces
        parts = [p.strip() for p in run_name.split(",")]

        # Sort for consistency
        parts.sort()

        clean_parts = []
        for p in parts:
            if "=" not in p:
                continue  # skip malformed parts
            key, val = p.split("=", 1)
            # Strip file extension if present
            val = os.path.splitext(val)[0]
            # Replace unwanted characters
            val = re.sub(r"[^\w.\-]", "_", val)
            clean_parts.append(f"{key}-{val}")

        dir_name = "_".join(clean_parts)
        return os.path.join(base_dir, dir_name)

    def _prepare_output_dir(self) -> str:
        base = self.config["model_dir"]
        subdir = self._run_name_to_dir(self.wandb_run.name)
        out_path = os.path.join(base, subdir)
        os.makedirs(out_path, exist_ok=True)
        log.info(
            "train_samples=%s, val_samples=%s, saving model to %s",
            self.config['train_samples'],
            self.config['val_samples'],
            out_path
        )
        self.config["out_path"] = out_path
        return out_path

    def _create_trainer(self) -> L.Trainer:
        profiler = None
        if self.config.get("do_profile", False):
            profiler = PyTorchProfiler(
                record_module_names=True,
                export_to_chrome=True,
                profile_memory=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self.config.get("profile_log_dir", "./logs")
                )
            )

        callbacks = []
        if self.config.get("use_early_stopping", False):
            callbacks.append(EarlyStopping(
                    monitor=f"val_{self.config['loss_fn_switch']}",
                    mode="min",
                    patience=self.config.get("patience", 10)
                ))
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        for cb in callbacks:
            print(cb, type(cb))
        trainer = L.Trainer(
            max_epochs=self.config["epochs"] if not self.config.get("debug", False) else 2,
            logger=self.logger,
            log_every_n_steps=self.config.get("log_every_n_steps", 1),
            enable_checkpointing=self.config.get("enable_checkpointing", False),
            precision=16 if self.device == "gpu" else 32,
            callbacks=callbacks,
            profiler=profiler,
            accelerator="gpu",
            devices=self.config.get("devices", 1)
        )
        return trainer

    def _read_max_lr_from_finder(self, lr_finder):
        lrs   = np.array(lr_finder.results["lr"])
        loss  = np.array(lr_finder.results["loss"])


        # replicate Lightning's notion of "diverging"
        best_so_far = np.minimum.accumulate(loss)
        thresh = 4.0  # match early_stop_threshold you used
        diverge = loss > thresh * best_so_far

        # index of first divergence; if none, fall back to last point
        idx = np.argmax(diverge)
        if not diverge.any() or idx == 0:
            max_stable_lr = lrs[-1]
        else:
            max_stable_lr = lrs[idx-1]

        return max_stable_lr.item()

    def _report_mem(self, msg=""):
        print(f"{msg} — allocated: "
              f"{torch.cuda.memory_allocated() / 1e9:.2f} GB, "
              f"reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    def tune_component_lr(self,
            model: L.LightningModule,
            key: str,
            trainer: L.Trainer,
            train_dl: torch.utils.data.DataLoader,
            steps: int = 100,
            early_stop: Optional[float] = None,
            use_lr_find: bool = True,
            suggestion_scale_factor: float = None
    ) -> float:
        tuner = Tuner(trainer)

        attr = f"{key}_lr"
        finder = tuner.lr_find(
            model,
            train_dataloaders=train_dl,
            num_training=steps,
            early_stop_threshold=early_stop,
            attr_name=attr
        )
        suggestion = finder.suggestion()
        if suggestion is None:
            raise ValueError(f"no learning rate found for {g}")
        if use_lr_find:
            if suggestion_scale_factor:
                suggestion *= suggestion_scale_factor
            return suggestion
        max_lr = self._read_max_lr_from_finder(finder)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return max_lr + suggestion / 4

    def tune_learning_rate(
            self,
            model: L.LightningModule,
            train_loader: torch.utils.data.DataLoader,
            fixed_lr: float = -1,
            use_lr_find: bool = True
    ) -> Dict[str, float]:
        freeze_encoder = bool(self.config.get("freeze_encoder", False))
        steps = int(self.config.get("lr_find_steps", 300))
        early_stop = self.config.get("early_stop_threshold", 4.0)
        debug = self.config.get("debug")

        target_keys: List[str] = []
        if not freeze_encoder:
            target_keys.append("encoder")
        for g in self.config["genes"]:
            target_keys.append(g)

        if fixed_lr != -1.0:
            return {k: fixed_lr for k in target_keys}
        if debug:
            return {k: 0.01 for k in target_keys}

        tmp_trainer = L.Trainer(
            accelerator="auto",
            devices=self.config.get("devices", 1),
            max_epochs=1,
            logger=False,
            enable_checkpointing=False
        )

        base_state = model.cpu().state_dict(keep_vars=True)
        orig_requires = [p.requires_grad for p in model.parameters()]

        tuned_max_lrs: Dict[str, float] = {}

        for key in target_keys:
            model.load_state_dict(base_state)
            for p in model.parameters():
                p.requires_grad = False
            for p in getattr(model, key).parameters():
                p.requires_grad = True

            tuned_max_lrs[key] = self.tune_component_lr(model, key, tmp_trainer, train_loader, steps, early_stop, use_lr_find)

            if self.wandb_run is not None:
                run = cast(Run, self.wandb_run)
                run.log(data={f"tuned_lr/{key}": tuned_max_lrs[key]})


            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # restore model to train state
        model.load_state_dict(base_state)
        for p, req in zip(model.parameters(), orig_requires):
            p.requires_grad = req

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._report_mem(f"finished tuning")
        return tuned_max_lrs

    def run(self):
        self._save_summary("started")
        data_module = get_data_module(self.config)
        data_module.setup("fit")
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        self.config["out_path"] = self.out_path
        model = get_model(self.config)

        # learning rate tuning
        self.config.setdefault("learning_rate", 1e-3) # init learning rate

        # store all results here
        fixed = self.config.get("global_fix_learning_rate", -1)
        use_lr_find = self.config.get("use_lr_find", False)
        if use_lr_find:
            suggestion_scale_factor = self.config.get("suggestion_scale_factor", 1.0)
        else:
            suggestion_scale_factor = None
        # build lr dict from either tuning or fixed if provided
        lrs = self.tune_learning_rate(model, train_loader, fixed, use_lr_find)

        print(f"Tuned learning rates: {lrs}")
        model.update_lr(lrs)
        if self.is_online:
            self.wandb_run.log({"tuned_lr": lrs})

        trainer = self._create_trainer()
        trainer.fit(model, train_loader, val_loader)

        if self.config.get("test_samples"):
            trainer.test(
                model=model,
                dataloaders=data_module.test_dataloader()
            )

        data_module.free_memory()
        self._save_summary("finished")
        log.info("Finished training")

        if self.config.get("spatial_plots", False):
            # Build a spatial dataset (no dataloader: we want per-row tile paths)
            use_test = self.config.get("test_samples", False)
            spatial_ds = get_spatial_dataset(
                data_dir=self.config["data_dir"],
                genes=self.config["genes"],
                samples=self.config["test_samples"] if use_test else self.config["val_samples"]
            )

            # Choose device and eval mode
            device = (
                    self.config.get("device")
                    or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            )
            model.eval()
            model.to(device)

            with torch.no_grad():
                # map genes to output indices once
                try:
                    gene_to_idx = {g: spatial_ds.genes.index(g) for g in self.config["genes"]}
                except ValueError as e:
                    raise ValueError(f"gene not found in spatial_plots in run of TrainerPipeline: {e}")

                for i in range(len(spatial_ds)):
                    # NOTE: STSpatialDataset.__getitem__ returns: img, target, x_t, y_t and maybe patient if return_patient=True
                    img_i, target_i, x_t_i, y_t_i = spatial_ds[i]

                    # Shape to (1, C, H, W) and send to device
                    if isinstance(img_i, np.ndarray):
                        img_t = torch.from_numpy(img_i)
                    else:
                        img_t = img_i
                    if img_t.dim() == 3:
                        img_t = img_t.unsqueeze(0)
                    img_t = img_t.to(device)

                    out = model(img_t)
                    if isinstance(out, (list, tuple)):
                        out = out[0]  # assume dim 0 is batch dim
                    out = torch.as_tensor(out)

                    # Make a 1D vector for the current tile's predictions
                    if out.dim() > 1:
                        out_vec = out[0].detach().cpu().flatten().numpy()
                    else:
                        out_vec = out.detach().cpu().flatten().numpy()

                    # Update all pred_{gene} columns for this tile
                    tile_path = spatial_ds.df.iloc[i]["tile"]
                    for g in self.config["genes"]:
                        gi = gene_to_idx[g]
                        if gi >= len(out_vec):
                            raise ValueError(
                                f"Model output size ({len(out_vec)}) smaller than index for gene {g} ({gi}).")
                        pred_val = float(out_vec[gi])
                        spatial_ds.add_result_for_tile(tile_path, pred_val, column=f"pred_{g}")

            for gene in self.config["genes"]:
                pred_col = f"pred_{gene}"

                df = spatial_ds.df  # includes new prediction columns
                if pred_col not in df.columns:
                    raise RuntimeError(f"Prediction column {pred_col!r} not found in spatial dataframe.")

                # Patients loop
                for patient, sub in df.groupby("patient"):
                    if sub.empty:
                        raise ValueError(f"no data for patient {patient}")

                    x = sub["x"].to_numpy(dtype=float)
                    y = sub["y"].to_numpy(dtype=float)
                    y_label = sub[gene].to_numpy(dtype=float)  # labels always present
                    y_pred = sub[pred_col].to_numpy(dtype=float)  # already filled in the NEW block above

                    if x.size == 0:
                        raise ValueError(f"no spatial data for patient {patient}")

                    # triptych == art of 3 images side by side
                    _plot_triptych_and_log(
                        x, y, y_label, y_pred,
                        patient=patient, gene=gene,
                        out_path=self.out_path,
                        is_online=self.is_online,
                        wandb_run=self.wandb_run if self.is_online else None,
                    )

    def _save_summary(self, status):
        serializable_cfg = {k: v for k, v in self.config.items() \
                            if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
        summary = serializable_cfg
        summary['status'] = status
        content = json.dumps(summary, indent=2)
        path = os.path.join(self.out_path, "config")
        with open(path, 'w') as f:
            f.write(content)
