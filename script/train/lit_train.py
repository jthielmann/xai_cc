# lit_training.py: Defines the core training pipeline and utilities for
# setting up, running, and summarizing model training with Lightning and W&B.

# Standard library imports
import os
import json
import logging
from pathlib import Path

# Third-party imports
import torch
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.callbacks import LearningRateMonitor

import itertools

# Local application imports
from script.model.lit_model import get_model
from script.data_processing.lit_STDataModule import get_data_module
from script.model.model_factory import get_encoder  # for encoder factory

# Prepare a module logger (configuration should be done in entry-point)
log = logging.getLogger(__name__)


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
                    "freeze_pretrained": "f_encdr",
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
            "loss_fn_switch", "encoder_type", "pretrained_out_dim"
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
            callbacks.append([EarlyStopping(
                    monitor=f"val_{self.config['loss_fn_switch']}",
                    mode="min",
                    patience=self.config.get("patience", 10)
                )])
        callbacks.append(LearningRateMonitor(logging_interval="step"))
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

    def tune_learning_rate(
        self,
        trainer: L.Trainer,
        model: L.LightningModule,
        train_loader: torch.utils.data.DataLoader,
    ) -> float:
        lr = self.config.get("fix_learning_Rate", None)
        if lr is None:
            if self.config.get("debug", False):
                lr = self.config.get("learning_rate", 1e-4)
                log.info("[LR-TUNER] Debug mode – using fixed lr = %.2e", lr)
            elif self.config["loss_fn_switch"] in {"WMSE", "weighted MSE"}:
                lr = self.config.get("learning_rate", 1e-3)  # don’t search
            else:
                tuner      = Tuner(trainer)
                lr_finder  = tuner.lr_find(
                    model,
                    train_dataloaders=train_loader,
                    num_training=self.config.get("lr_find_steps", 300),
                )
                lr = lr_finder.suggestion()
                if lr is None:
                    raise ValueError("lr_finder did not find a solution")
                log.info("[LR-TUNER] Suggested lr = %.2e", lr)

            model.update_lr(lr)
            self.config["learning_rate"] = lr

            if self.wandb_run is not None:
                self.wandb_run.log({"tuned_lr": lr})

        return lr

    def run(self):
        data_module = get_data_module(self.config)
        data_module.setup("fit")
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        self.config["out_dir"] = self.out_dir
        model = get_model(self.config)

        # learning rate tuning
        self.config.setdefault("learning_rate", 1e-3) # init learning rate
        param_grid = {
            "lr_find_steps": [1000],
            "early_stop_threshold": [None],
        }

        # prepare the trainer (we'll reuse it each run)
        tmp_trainer = L.Trainer(
            accelerator="gpu",
            devices=self.config.get("devices", 1),
            max_epochs=1,
            logger=False,
            enable_checkpointing=False
        )

        lr_finder = Tuner(tmp_trainer).lr_find(
            model, train_dataloaders=train_loader,
            num_training=self.config.get("lr_find_steps", 300),
            enable_checkpointing=False

        )
        # store all results here
        new_lr = self.config.get("fix_learing_rate", None)
        if new_lr is None:
            # loop over every combination of (steps, min_lr, max_lr, early_stop_threshold)
            if self.debug:
                new_lr = 0.01
            else:
                new_lr = lr_finder.suggestion()
            if new_lr is None:
                raise ValueError("lr_finder did not find a learning rate")
        print(f"Using learning rate: {new_lr}")
        model.update_lr(new_lr)
        self.config["learning_rate"] = new_lr
        if self.is_online:
            self.wandb_run.log({"tuned_lr": new_lr})

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
        """
        Dump a JSON summary of the final config (with status='completed')
        into out_dir/config.json for record-keeping.
        """
        serializable_cfg = {k: v for k, v in self.config.items() \
                            if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
        summary = serializable_cfg
        summary['status'] = 'completed'
        content = json.dumps(summary, indent=2)
        path = os.path.join(self.out_dir, "config.json")
        with open(path, 'w') as f:
            f.write(content)
