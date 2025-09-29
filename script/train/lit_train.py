# lit_training.py: Defines the core training pipeline and utilities for
# setting up, running, and summarizing model training with Lightning and W&B.
import gc
# Standard library imports
import os
import json
import logging

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
from typing import Dict, List
from pathlib import Path
from contextlib import contextmanager

# Local application imports
from script.model.lit_model import get_model
from script.main_utils import make_run_name_from_config
from script.data_processing.lit_STDataModule import get_data_module
from script.data_processing.data_loader import get_spatial_dataset
# Prepare a module logger (configuration should be done in entry-point)
log = logging.getLogger(__name__)
from lightning.pytorch import seed_everything
import os, json, pandas as pd
from collections import Counter
from script.configs.normalization import resolve_norm


@contextmanager
def _temp_cwd(path: str | os.PathLike):
    """Temporarily change the current working directory.

    Used to redirect Lightning's lr_find temporary checkpoint ('.lr_find_*.ckpt')
    into a dedicated dump folder.
    """
    old = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)

def _validate_config_and_shapes(cfg, model, loader):
    # 1. Gene count matches model output
    dummy_batch = next(iter(loader))
    x, y = dummy_batch[:2]   # first two items are usually (img, expr)
    y_hat = model(x)
    if y_hat.shape[1] != len(cfg["genes"]):
        raise ValueError(f"Model outputs {y_hat.shape[1]} genes, "
                         f"but config['genes'] has {len(cfg['genes'])}")

    # 2. Shapes align
    if y_hat.shape != y.shape:
        raise ValueError(f"Output shape {y_hat.shape} != target shape {y.shape}")

    # 3. Splits disjoint
    all_sets = [set(cfg.get(k, [])) for k in ("train_samples","val_samples","test_samples")]
    overlap = set.intersection(*all_sets) if all_sets else set()
    if overlap:
        raise ValueError(f"Patient overlap across splits: {overlap}")

def _verify_log_frozen(model, freeze_encoder, wandb_run=None):
    # strict encoder check + counts
    enc_trainable_names, enc_frozen_names = [], []
    if hasattr(model, "encoder"):
        enc_trainable_names = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
        enc_frozen_names    = [n for n, p in model.encoder.named_parameters() if not p.requires_grad]
        if freeze_encoder and enc_trainable_names:
            raise RuntimeError(f"freeze_encoder=True but trainable: {enc_trainable_names[:5]}...")
        if not freeze_encoder and not enc_trainable_names:
            raise RuntimeError("freeze_encoder=False but encoder has no trainable params!")

    # totals
    total_params   = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params  = total_params - trainable_params

    # per-top-module breakdown
    mods = {}
    for name, p in model.named_parameters():
        top = name.split('.', 1)[0]
        t, tr = mods.get(top, (0, 0))
        n = p.numel()
        mods[top] = (t + n, tr + (n if p.requires_grad else 0))

    # log.info summary
    pct = 100.0 * trainable_params / total_params if total_params else 0.0
    log.info(f"[verify] freeze_encoder={freeze_encoder} | trainable {trainable_params}/{total_params} ({pct:.1f}%)")
    for m, (t, tr) in mods.items():
        mpct = 100.0 * tr / t if t else 0.0
        log.info(f"  - {m:12s} {tr}/{t} ({mpct:.1f}%)")

    # wandb logging (single block)
    if wandb_run:
        log_dict = {
            "params/total": total_params,
            "params/trainable": trainable_params,
            "params/frozen": frozen_params,
            "config/freeze_encoder": freeze_encoder,
        }
        if hasattr(model, "encoder"):
            enc_trainable = sum(p.numel() for _, p in model.encoder.named_parameters() if p.requires_grad)
            enc_frozen    = sum(p.numel() for _, p in model.encoder.named_parameters() if not p.requires_grad)
            log_dict.update({
                "encoder/params_trainable": enc_trainable,
                "encoder/params_frozen": enc_frozen,
            })
        wandb_run.log(log_dict)
        for m, (t, tr) in mods.items():
            wandb_run.log({
                f"{m}/total": t,
                f"{m}/trainable": tr,
                f"{m}/frozen": t - tr,
                f"{m}/trainable_pct": (100.0 * tr / t) if t else 0.0
            })



def _log_dataset_info(cfg, out_dir, train=None, val=None, test=None):
    def split_df(loader, split):
        ds = getattr(loader, "dataset", loader)
        pats = getattr(ds, "patient_per_item", None) or getattr(ds, "patients", None)
        if pats is None and hasattr(ds, "df") and "patient" in ds.df: pats = ds.df["patient"].tolist()
        if pats is None: pats = ["unknown"] * len(ds)
        c = Counter(pats)
        return pd.DataFrame({"split": split, "patient": list(c), "n_items": list(c.values())})

    dfs = []
    if train: dfs.append(split_df(train, "train"))
    if val:   dfs.append(split_df(val, "val"))
    if test:  dfs.append(split_df(test, "test"))
    manifest = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    meta = {
        "data_dir": cfg.get("data_dir",""),
        "n_train_samples": len(cfg.get("train_samples",[])),
        "n_val_samples":   len(cfg.get("val_samples",[])),
        "n_test_samples":  len(cfg.get("test_samples",[])),
        "n_patients": manifest.patient.nunique() if not manifest.empty else 0,
        "n_genes": len(cfg.get("genes",[])),
        "encoder_type": cfg.get("encoder_type",""),
    }

    os.makedirs(out_dir, exist_ok=True)
    manifest.to_csv(os.path.join(out_dir,"split_manifest.csv"), index=False)
    open(os.path.join(out_dir,"genes.txt"),"w").write("\n".join(cfg.get("genes",[])))
    open(os.path.join(out_dir,"dataset_meta.json"),"w").write(json.dumps(meta,indent=2))

    log.info("[dataset] %s", meta)
    if not manifest.empty: log.info("[splits]\n%s", manifest.head())

def _save_spatial_parquets(spatial_df: pd.DataFrame, genes: list[str], out_dir: str) -> None:
    base = os.path.join(out_dir, "spatial_parquet")
    os.makedirs(base, exist_ok=True)
    idx = []

    for gene in genes:
        pred_col = f"pred_{gene}"
        if pred_col not in spatial_df.columns:
            raise RuntimeError(f"Missing column {pred_col} in spatial_df.")
        for patient, sub in spatial_df.groupby("patient"):
            if sub.empty:
                continue
            df = pd.DataFrame({
                "x":     sub["x"].astype("float32"),
                "y":     sub["y"].astype("float32"),
                "label": sub[gene].astype("float32"),
                "pred":  sub[pred_col].astype("float32"),
            })
            df["diff"] = (df["pred"] - df["label"]).astype("float32")

            fn = f"{patient}__{gene}.parquet"
            fp = os.path.join(base, fn)
            df.to_parquet(fp, index=False)  # uses pyarrow if installed
            idx.append({"patient": patient, "gene": gene, "path": fp, "n": len(df)})

    if idx:
        pd.DataFrame(idx).to_parquet(os.path.join(base, "_index.parquet"), index=False)

def _determine_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
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

def _cuda_supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        # Available in recent PyTorch versions; guards for runtime envs
        return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        pass
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8  # Ampere (A100/3090) and newer (Ada 4090+)
    except Exception:
        return False


def _choose_precision() -> str:
    """Return a Lightning precision string based on runtime support."""
    if torch.cuda.is_available():
        return "bf16-mixed" if _cuda_supports_bf16() else "16-mixed"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # MPS benefits from fp16 mixed; bf16 not broadly supported/stable
        return "16-mixed"
    return "32"

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

        # Resolve a central dump directory for incidental outputs (logs, temps)
        try:
            # Prefer explicit config, then env, then repo-root/dump
            repo_root = Path(__file__).resolve().parents[2]
        except Exception:
            repo_root = Path.cwd()
        self.dump_dir = str(
            Path(
                self.config.get("dump_dir")
                or os.environ.get("XAI_DUMP_DIR")
                or (repo_root / "dump")
            ).resolve()
        )
        os.makedirs(self.dump_dir, exist_ok=True)

        self.is_sweep = hasattr(self.wandb_run.config, "sweep_parameter_names")

        self.is_online = self.config.get("log_to_wandb")
        if not self.is_online:
            self.project = "local_run"
            self.logger = None
        else: # online logging to wandb
            self.project   = self.wandb_run.project
            if self.is_sweep:
                # Build compact name from sweep hyperparameters
                param_names = list(getattr(self.wandb_run.config, "sweep_parameter_names", []))
                name = make_run_name_from_config(self.config, param_names).replace("-", ", ")
            else:
                name = self.config["project"] + " " + self.config["name"]
            gid = self.config.get("genes_id")
            if gid and "genes_id=" not in name:
                name = f"{name}, genes_id={gid}"
            self.wandb_run.name = name
            self.wandb_run.notes = f"Training {self.wandb_run.name} on {self.wandb_run.project}"
            # Reuse the existing W&B run created in the entrypoint (sweeps or single runs)
            # to avoid spawning a second run with its own config.
            self.logger = WandbLogger(
                project=self.wandb_run.project,
                name=name,
                experiment=self.wandb_run,
                save_dir=self.dump_dir,
            )



        # Ensure a project root (sweep root) exists for project-specific outputs
        try:
            if not self.config.get("sweep_dir"):
                proj = self.config.get("project", "project")
                self.config["sweep_dir"] = os.path.join(self.config["model_dir"], proj)
            os.makedirs(self.config["sweep_dir"], exist_ok=True)
        except Exception:
            pass

        self.device  = _determine_device()
        self.out_path = self._prepare_output_dir()
        # Normalize: capture stats without logging panels to W&B
        try:
            stats = resolve_norm(self.config.get("encoder_type", ""))
            norm_meta = {
                "mode": "encoder",
                "encoder_type": self.config.get("encoder_type", ""),
                "mean": list(stats.mean),
                "std": list(stats.std),
            }
            # expose mean/std in config so results CSV can include them (cfg_normalize_mean/std)
            self.config["normalize_mean"] = norm_meta["mean"]
            self.config["normalize_std"] = norm_meta["std"]
            # still save to run folder for traceability
            os.makedirs(self.out_path, exist_ok=True)
            with open(os.path.join(self.out_path, "normalization.json"), "w") as f:
                json.dump(norm_meta, f, indent=2)
            # intentionally do NOT log normalize/* to wandb
        except Exception:
            pass

    def _run_name_to_dir(self, run_name: str) -> str:
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
        return dir_name

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
                    self.config.get("profile_log_dir", os.path.join(self.dump_dir, "profiles"))
                )
            )

        callbacks = []
        if self.config.get("use_early_stopping", False):
            callbacks.append(EarlyStopping(
                    monitor=f"val_{self.config['loss_fn_switch']}",
                    mode="min",
                    patience=self.config.get("patience", 10)
                ))
        if self.config.get("monitorLr", False):
            callbacks.append(LearningRateMonitor(logging_interval="step"))
        for cb in callbacks:
            log.info(cb, type(cb))

        accelerator = "auto"
        devices = self.config.get("devices", "auto")

        strategy = self.config.get("strategy", None)
        if strategy is None and (devices == "auto" or (isinstance(devices, int) and devices > 1)) and torch.cuda.is_available():
            strategy = "ddp_find_unused_parameters_false"
        else:
            strategy = "auto"
        precision = _choose_precision()
        trainer = L.Trainer(
            max_epochs=self.config["epochs"] if not self.config.get("debug", False) else 2,
            logger=self.logger,
            log_every_n_steps=self.config.get("log_every_n_steps", 1),
            enable_checkpointing=self.config.get("enable_checkpointing", False),
            precision=precision,
            callbacks=callbacks,
            profiler=profiler,
            accelerator=accelerator,
            devices=self.config.get("devices", 1),
            deterministic=True,
            strategy=strategy,
            default_root_dir=self.dump_dir,
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
        if not self.device == "cuda":
            return
        log.info(f"{msg} — allocated: "
              f"{torch.cuda.memory_allocated() / 1e9:.2f} GB, "
              f"reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    def tune_component_lr(self,
            model: L.LightningModule,
            key: str,
            trainer: L.Trainer,
            train_dl: torch.utils.data.DataLoader,
            steps: int = 100,
            early_stop: Optional[float] = None,
            use_lr_find: bool = True
    ) -> float:
        tuner = Tuner(trainer)

        attr = f"{key}_lr"
        # Redirect lr_find temporary checkpoint into dump_dir
        with _temp_cwd(self.dump_dir):
            finder = tuner.lr_find(
                model,
                train_dataloaders=train_dl,
                num_training=steps,
                early_stop_threshold=early_stop,
                attr_name=attr
            )
        suggestion = finder.suggestion()
        if suggestion is None:
            raise ValueError(f"no learning rate found for {key}")
        if use_lr_find:
            suggestion_scale_factor = self.config.get("suggestion_scale_factor", 1.0)
            suggestion *= suggestion_scale_factor
            return suggestion
        max_lr = self._read_max_lr_from_finder(finder)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return max_lr + suggestion / 4

    def _log_batch_stats(self, loader: torch.utils.data.DataLoader, split: str) -> None:
        """Log min/max/mean/std (global and per-channel) for a single batch after transforms.

        Best-effort: swallow exceptions so training is not blocked if a loader is empty or custom.
        """
        if not loader:
            return
        try:
            batch = next(iter(loader))
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if not torch.is_tensor(x):
                return
            x = x.detach().float()
            shape = tuple(x.shape)
            gmin = float(x.min())
            gmax = float(x.max())
            gmean = float(x.mean())
            gstd = float(x.std(unbiased=False))
            log.info(f"[stats] {split}: shape={shape} dtype={x.dtype} min={gmin:.4f} max={gmax:.4f} mean={gmean:.4f} std={gstd:.4f}")
            if x.dim() == 4:  # (B,C,H,W)
                ch_mean = x.mean(dim=(0, 2, 3))
                ch_std  = x.std(dim=(0, 2, 3), unbiased=False)
                log.info(
                    f"[stats] {split}: per_channel_mean={[round(float(m),4) for m in ch_mean]} "
                    f"per_channel_std={[round(float(s),4) for s in ch_std]}"
                )
        except Exception as e:
            try:
                log.debug(f"[stats] Unable to log {split} batch stats: {e}")
            except Exception:
                pass

    def _generate_spatial_plots(self, model):
        # Build a spatial dataset (no dataloader: we want per-row tile paths)
        use_test = self.config.get("test_samples", False)
        spatial_ds = get_spatial_dataset(
            data_dir=self.config["data_dir"],
            genes=self.config["genes"],
            samples=self.config["test_samples"] if use_test else self.config["val_samples"],
            meta_data_dir=self.config.get('meta_data_dir', '/meta_data/')
        )

        # Choose device and eval mode
        device = _determine_device()
        model.eval()
        model.to(device)

        with torch.no_grad():
            # map genes to output indices once
            try:
                gene_to_idx = {g: model.gene_to_idx[g] for g in self.config["genes"]}
            except ValueError as e:
                raise ValueError(f"gene not found in spatial_plots in run of TrainerPipeline: {e}")

            for i in range(len(spatial_ds)):
                # NOTE: STSpatialDataset.__getitem__ returns: img, target, x_t, y_t and maybe patient if return_patient=True
                img_i, *_ = spatial_ds[i]

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
        _save_spatial_parquets(spatial_ds.df, self.config["genes"], self.out_path)


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

            if self.wandb_run is not None and self.config.get("log_lr", False):
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
        seed_everything(42, workers=True)
        self._save_summary("started")
        data_module = get_data_module(self.config)
        try:
            data_module.setup("fit")
            train_loader = data_module.train_dataloader()
            val_loader = data_module.val_dataloader()
            if self.config.get("test_samples"):
                test_loader = data_module.test_dataloader()
            else:
                test_loader = None
            # Log batch stats after transforms for sanity (min/max/mean/std)
            if self.config.get("log_batch_stats", True):
                self._log_batch_stats(train_loader, "train")
                if val_loader:
                    self._log_batch_stats(val_loader, "val")
                if test_loader:
                    self._log_batch_stats(test_loader, "test")
            _log_dataset_info(self.config, self.out_path,
                             train=train_loader, val=val_loader,
                             test=test_loader if self.config.get("test_samples") else None)
            self.config["out_path"] = self.out_path
            model = get_model(self.config)

            # better save than sorry, also count and log trainable param
            _verify_log_frozen(model, self.config.get("freeze_encoder", False),
                                   self.wandb_run if self.is_online and
                                   self.config.get("verify_log_frozen", False) else None)
            with torch.inference_mode():
                _validate_config_and_shapes(self.config, model, train_loader)
            # learning rate tuning
            self.config.setdefault("learning_rate", 1e-3) # init learning rate

            # store all results here
            fixed = self.config.get("global_fix_learning_rate", -1)
            use_lr_find = self.config.get("use_lr_find", False)
            # build lr dict from either tuning or fixed if provided
            lrs = self.tune_learning_rate(model, train_loader, fixed, use_lr_find)

            log.info("Tuned learning rates: %s", lrs)
            model.update_lr(lrs)
            if self.is_online:
                self.wandb_run.summary.update({"tuned_lr": lrs})

            trainer = self._create_trainer()
            trainer.fit(model, train_loader, val_loader)

            if self.config.get("test_samples"):
                trainer.test(
                    model=model,
                    dataloaders=test_loader
                )
        finally:
            data_module.free_memory()
            self._save_summary("finished")
            log.info("Finished training")

        if self.config.get("spatial_plots", False):
            self._generate_spatial_plots(model)

    def _save_summary(self, status):
        serializable_cfg = {k: v for k, v in self.config.items() \
                            if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
        summary = serializable_cfg
        summary['status'] = status
        content = json.dumps(summary, indent=2)
        path = os.path.join(self.out_path, "config")
        try:
            with open(path, 'w') as f:
                f.write(content)
        except OSError as e:
            # Fallback to dump_dir if writing to out_path fails (e.g., quota)
            try:
                os.makedirs(self.dump_dir, exist_ok=True)
                alt = os.path.join(self.dump_dir, f"config_fallback.json")
                with open(alt, 'w') as f:
                    f.write(content)
                log.warning("Failed to write summary to %s (%s). Wrote fallback to %s", path, e, alt)
            except Exception:
                # Last resort: ignore to avoid training crash
                log.error("Failed to persist summary to both %s and dump_dir: %s", path, e)
