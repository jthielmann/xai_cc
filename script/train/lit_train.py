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
from wandb.wandb_run import Run
from typing import Optional, cast, Any
import numpy as np
import torch.nn as nn
from typing import Dict, List
import itertools

# Local application imports
from script.model.lit_model import get_model
from script.data_processing.lit_STDataModule import get_data_module
from script.model.model_factory import get_encoder  # for encoder factory

# Prepare a module logger (configuration should be done in entry-point)
log = logging.getLogger(__name__)


class SAEOnlyModule(L.LightningModule):
    """
    LightningModule for pre-training a single autoencoder variant.
    """
    def __init__(self, encoder: nn.Module, autoencoder: nn.Module, lr: float):
        super().__init__()
        self.encoder = encoder
        self.autoencoder = autoencoder
        self.lr = lr
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        z = self.encoder(x)
        return self.autoencoder(z)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        z = self.encoder(x)
        z_hat = self.autoencoder(z)
        loss = self.loss_fn(z_hat, z.detach())
        self.log('ae_train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        z = self.encoder(x)
        z_hat = self.autoencoder(z)
        loss = self.loss_fn(z_hat, z.detach())
        self.log('ae_val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.autoencoder.parameters(), lr=self.lr)

    def set_num_training_batches(self, n: int):
        # stub for compatibility with OneCycleLR if needed
        pass


def train(config: dict, run: wandb.sdk.wandb_run.Run):
    """
    Simple entrypoint if you want to train by calling train(config).
    Instantiates TrainerPipeline and runs the full training process.
    """
    pipeline = TrainerPipeline(config, run)
    pipeline.run()


def get_trainer(cfg: dict, logger: WandbLogger) -> L.Trainer:
    """
    Build and return a Lightning Trainer configured with:
      - max_epochs, logging, checkpointing, precision,
      - optional profiling, early-stopping,
      - GPU accelerator and device count.
    """
    profiler = None
    if cfg.get("do_profile", False):
        # Enable Chrome trace profiling if requested
        profiler = PyTorchProfiler(
            record_module_names=True,
            export_to_chrome=True,
            profile_memory=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                cfg.get("profile_log_dir", "./logs")
            )
        )

    trainer = L.Trainer(
        max_epochs=cfg["epochs"],
        logger=logger,
        log_every_n_steps=cfg.get("log_every_n_steps", 1),
        enable_checkpointing=cfg.get("enable_checkpointing", False),
        default_root_dir="/tmp/lr_finder",
        precision=16,
        callbacks=[EarlyStopping(
            monitor=f"val_{cfg['loss_fn_switch']}",
            mode="min",
            patience=cfg.get("patience", 10)
        )] if cfg.get("use_early_stopping", False) else [],
        profiler=profiler,
        accelerator="gpu",
        devices=cfg.get("devices", 1)
    )
    return trainer


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


class TrainerPipeline:
    def __init__(self, config: dict, run: wandb.sdk.wandb_run.Run):
        self.wandb_run = run
        self.config    = config
        self.debug     = self.config.get("debug")

        self.is_sweep = hasattr(self.wandb_run.config, "sweep_parameter_names")

        self.is_online = self.config.get("log_to_wandb")
        if self.is_online:
            self.project   = self.wandb_run.project
            if self.is_sweep:
                ABBR = {
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
                    short = ABBR.get(k, k)
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
        else:
            self.project = "local_run"
            self.logger = None
        required = [
            "genes", "train_samples", "val_samples",
            "data_dir", "batch_size", "epochs",
            "loss_fn_switch", "encoder_type"
        ]
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"Missing config keys: {missing}")

        self.device  = _determine_device()
        self.out_dir = self._prepare_output_dir()

    def _prepare_output_dir(self) -> str:
        base = Path("..") / "models" / self.project
        subdir = self.config.get("subdir")
        if subdir is None:
            subdir = self.wandb_run.name
        out_dir = os.path.join(base, subdir)
        os.makedirs(out_dir, exist_ok=True)
        log.info(
            "train_samples=%s, val_samples=%s, saving model to %s",
            self.config['train_samples'],
            self.config['val_samples'],
            out_dir
        )
        return out_dir

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

    def report_mem(self, msg=""):
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
            use_lr_find: bool = True,
            suggestion_scale_factor: float = None
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
        self.report_mem(f"finished tuning")
        return tuned_max_lrs

    def run(self):
        data_module = get_data_module(self.config)
        data_module.setup("fit")
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        self.config["out_dir"] = self.out_dir
        model = get_model(self.config)

        # learning rate tuning
        self.config.setdefault("learning_rate", 1e-3) # init learning rate

        # store all results here
        fixed = self.config.get("global_fix_learning_rate", -1)
        use_lr_find = self.config.get("use_lr_find", True)
        if use_lr_find:
            suggestion_scale_factor = self.config.get("suggestion_scale_factor", 1.0)
        else:
            suggestion_scale_factor = None
        # build lr dict from either tuning or fixed if provided
        lrs = self.tune_learning_rate(model, train_loader, fixed, use_lr_find, suggestion_scale_factor)

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
        self._save_summary()
        log.info("Finished training")

    def _save_summary(self):
        serializable_cfg = {k: v for k, v in self.config.items() \
                            if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
        summary = serializable_cfg
        summary['status'] = 'completed'
        content = json.dumps(summary, indent=2)
        path = os.path.join(self.out_dir, "config.json")
        with open(path, 'w') as f:
            f.write(content)
