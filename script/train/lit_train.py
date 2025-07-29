# lit_training.py: Defines the core training pipeline and utilities for
# setting up, running, and summarizing model training with Lightning and W&B.

# Standard library imports
import os
import json
import logging
from pathlib import Path

# Third-party imports
import torch
import torch.nn as nn
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.tuner.tuning import Tuner

# Local application imports
from script.model.lit_model import get_model
from script.data_processing.lit_STDataModule import get_data_module
from script.model.model_factory import get_encoder  # for encoder factory
from script.model.lit_ae import get_autoencoder     # AE factory

# Prepare a module logger
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
        self.is_online = self.config.get("log_to_wandb", False)

        if self.is_online:
            self.project = self.wandb_run.project
            if self.is_sweep:
                ABBR = {
                    "learning_rate": "lr",
                    "batch_size": "bs",
                    # ...
                }
                parts = []
                for k in self.wandb_run.config.sweep_parameter_names:
                    short = ABBR.get(k, k)
                    val = self.config.get(k)
                    parts.append(f"{short}={val}")
                name = ", ".join(parts)
            else:
                name = f"{self.config['project']} {self.config['name']}"
            self.wandb_run.name = name
            self.wandb_run.notes = f"Training {name} on {self.wandb_run.project}"
            self.logger = WandbLogger(project=self.wandb_run.project, name=name)
        else:
            self.project = "local_run"
            self.logger  = None

        required = [
            "genes", "train_samples", "val_samples",
            "data_dir", "batch_size", "epochs",
            "loss_fn_switch", "encoder_type", "encoder_out_dim"
        ]
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"Missing config keys: {missing}")

        self.device  = _determine_device()
        self.out_dir = self._prepare_output_dir()

    def _prepare_output_dir(self) -> str:
        base = Path("..") / "models" / self.project
        subdir = self.config.get("subdir", self.wandb_run.name)
        out_dir = base / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "train_samples=%s, val_samples=%s, saving model to %s",
            self.config['train_samples'],
            self.config['val_samples'],
            out_dir
        )
        return str(out_dir)

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
            callbacks.append(
                EarlyStopping(
                    monitor=f"val_{self.config['loss_fn_switch']}",
                    mode="min",
                    patience=self.config.get("patience", 10)
                )
            )
        callbacks.append(
            LearningRateMonitor(logging_interval="step", log_momentum=False)
        )

        return L.Trainer(
            accelerator="gpu",
            devices=self.config.get("devices", 1),
            max_epochs=self.config["epochs"] if not self.debug else 2,
            logger=self.logger,
            log_every_n_steps=self.config.get("log_every_n_steps", 1),
            enable_checkpointing=self.config.get("enable_checkpointing", False),
            precision=16 if self.device == "gpu" else 32,
            callbacks=callbacks,
            profiler=profiler
        )

    def run(self):
        # --- data & model setup ---
        dm = get_data_module(self.config)
        dm.setup("fit")
        train_loader = dm.train_dataloader()
        val_loader   = dm.val_dataloader()
        if self.config.get("test_samples"):
            test_loader  = dm.test_dataloader()
        num_batches = len(train_loader)
        if self.config.get("fix_learing_rate", False):
            lr = self.config.get("fix_learing_rate")
            self.config["learning_rate"] = lr
        else:
            tmp_trainer = L.Trainer(
                accelerator="gpu",
                devices=self.config.get("devices", 1),
                max_epochs=1,
                logger=False,
                enable_checkpointing=False
            )
            lr_finder = Tuner(tmp_trainer).lr_find(
                get_model(self.config),
                train_dataloaders=train_loader,
                num_training=self.config.get("lr_find_steps", 300)
            )
            lr = self.config.get("fix_learing_rate", None)
            if lr is None:
                lr = 0.01 if self.debug else lr_finder.suggestion()
                if lr is None:
                    raise ValueError("lr_finder did not find a learning rate")
            self.config["learning_rate"] = lr
        print(f"Using learning rate: {lr}")
        if self.is_online:
            self.wandb_run.log({"tuned_lr": self.config["learning_rate"]})

        if self.config.get("pretrain_sae", False):
            for ae_type in self.config.get("ae_variants", ["sparse"]):
                # 1) Pre-train this AE-only module
                ae = get_autoencoder(config=self.config)
                ae_module = SAEOnlyModule(
                    encoder=get_encoder(self.config["encoder_type"]),
                    autoencoder=ae,
                    lr=self.config.get("ae_lr", 1e-3),
                )
                ae_module.set_num_training_batches(num_batches)
                trainer_ae = self._create_trainer()
                trainer_ae.max_epochs = self.config.get("sae_epochs", 5)
                trainer_ae.fit(ae_module, train_loader, val_loader)

                # save pretrained AE weights
                ckpt_path = os.path.join(self.out_dir, f"sae_pretrained_{ae_type}.pt")
                torch.save(ae_module.state_dict(), ckpt_path)

                # 2) Train full regression model with this AE
                model = get_model(self.config)
                model.set_num_training_batches(num_batches)
                model.sae_pre.load_state_dict(torch.load(ckpt_path, map_location=self.device))
                if self.config.get("freeze_sae_after_pretrain", True):
                    for p in model.sae_pre.parameters():
                        p.requires_grad = False

                trainer_reg = self._create_trainer()
                trainer_reg.fit(model, train_loader, val_loader)

        else:
            model = get_model(self.config)
            model.set_num_training_batches(num_batches)
            trainer = self._create_trainer()
            trainer.fit(model, train_loader, val_loader)

        if self.config.get("test_samples"):
            trainer.test(
                model=model,
                dataloaders=dm.test_dataloader()
            )

        dm.free_memory()
        self._save_summary()
        log.info("Finished training")

    def _save_summary(self):
        serializable = {
            k: v for k, v in self.config.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
        }
        serializable['status'] = 'completed'
        with open(os.path.join(self.out_dir, "config.json"), 'w') as f:
            json.dump(serializable, f, indent=2)