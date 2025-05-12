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
        precision=16,
        callbacks=[EarlyStopping(
            monitor=f"validation_{cfg['error_metric_name']}",
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
        # 1) store the passed-in run & config
        self.wandb_run = run
        self.config    = config
        self.project   = self.wandb_run.project
        name = ", ".join(f"{k}: {self.config.get(k)}" for k in self.wandb_run.config.sweep_parameter_names)
        self.wandb_run.name = name
        self.wandb_run.notes = f"Training {self.wandb_run.name} on {self.wandb_run.project}"
        # 2) build the Lightning logger off that run
        self.logger = WandbLogger(
            project=self.wandb_run.project,
            name=name
        )

        # 3) the rest of your setup
        required = [
            "genes", "train_samples", "val_samples",
            "data_dir", "batch_size", "epochs", "learning_rate",
            "error_metric_name", "encoder_type", "pretrained_out_dim"
        ]
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"Missing config keys: {missing}")

        self.device  = _determine_device()
        self.out_dir = self._prepare_output_dir()

    def _prepare_output_dir(self) -> str:
        """
        Create a directory for model checkpoints and summaries under:
          ../models/{project}/{subdir}
        Where subdir is either provided, 'lit_testing' in debug, or the run name.
        """
        base = Path("..") / "models" / self.project
        subdir = self.config.get("subdir")
        if subdir is None:
            subdir = "lit_testing" if self.config.get("debug", False) else self.wandb_run.name
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
        """
        Build and return a Lightning Trainer instance
        using the stored logger, callbacks, precision, and devices.
        """
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

        trainer = L.Trainer(
            max_epochs=self.config["epochs"] if not self.config.get("debug", False) else 2,
            logger=self.logger,
            log_every_n_steps=self.config.get("log_every_n_steps", 1),
            enable_checkpointing=self.config.get("enable_checkpointing", False),
            precision=16 if self.device == "gpu" else 32,
            callbacks=[EarlyStopping(
                monitor=f"validation_{self.config['error_metric_name']}",
                mode="min",
                patience=self.config.get("patience", 10)
            )] if self.config.get("use_early_stopping", False) else [],
            profiler=profiler,
            accelerator="gpu",
            devices=self.config.get("devices", 1)
        )
        return trainer

    def run(self):
        """
        Execute the training steps:
          1. Instantiate data module & loaders
          2. Build trainer & model
          3. Fit model (train/val)
          4. Optionally test on test set
          5. Cleanup and save summary JSON
        """
        # 1) Data
        data_module = get_data_module(self.config)
        data_module.setup("fit")
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        # 2) Trainer & Model
        trainer = self._create_trainer()
        model = get_model(self.config)
        model.set_num_training_batches(len(train_loader))

        # 3) Train
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )

        # 4) Test if provided
        if self.config.get("test_samples"):
            trainer.test(
                model=model,
                dataloaders=data_module.test_dataloader()
            )

        # 5) Cleanup & Save
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
