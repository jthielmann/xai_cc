import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from script.model.model_factory import get_encoder, infer_encoder_out_dim
from script.model.lit_ae import SparseAutoencoder
from script.data_processing.transforms import build_transforms
from script.data_processing.data_loader import (
    get_dataset,
    get_dataset_single_file,
)


def _unwrap_encoder_output(z: torch.Tensor) -> torch.Tensor:
    """Normalize encoder outputs to 2D (B, D) using consistent rules.

    - If HF-style output with last_hidden_state, use it
    - If 4D (B, C, H, W), flatten spatial dims
    - If 3D (B, T, D), mean-pool tokens → (B, D)
    - If tuple/list, use first tensor
    """
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
    return z


class SAEOnEncoder(L.LightningModule):
    """
    LightningModule that trains a SparseAutoencoder on top of a frozen encoder,
    reconstructing the encoder's 2D features.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(cfg)

        # Encoder (frozen)
        self.encoder = get_encoder(cfg["encoder_type"])  # expects valid local weights
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Infer feature dimension and build SAE
        d_in = infer_encoder_out_dim(self.encoder)
        sae_cfg = dict(cfg.get("sae", {}))
        sae_cfg.update({
            "d_in": int(d_in),
            # allow explicit top-level overrides for convenience
            "d_hidden": int(cfg.get("d_hidden", sae_cfg.get("d_hidden", 1024))),
            "k": int(cfg.get("k", sae_cfg.get("k", max(1, int(d_in // 16))))),
            "allow_negative_topk": bool(cfg.get("allow_negative_topk", sae_cfg.get("allow_negative_topk", False))),
        })
        self.sae = SparseAutoencoder(sae_cfg)

        self.loss_fn = nn.MSELoss()
        self.best_val = float("inf")
        self._sanity_skipped = False

    def train(self, mode: bool = True):
        super().train(mode)
        # keep encoder in eval to avoid BN running-stat drift
        try:
            self.encoder.eval()
            for m in self.encoder.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        except Exception:
            pass
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            z = self.encoder(x)
            z = _unwrap_encoder_output(z)
        # SAE reconstructs encoder features; gradients only through SAE
        recon = self.sae(z)
        return recon, z

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        recon, target = self(x)
        loss = self.loss_fn(recon, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        recon, target = self(x)
        loss = self.loss_fn(recon, target)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Skip sanity check epoch
        if not self._sanity_skipped:
            self._sanity_skipped = True
            return

        out_dir = self.hparams.get("out_dir") or self.hparams.get("out_path") or "."
        os.makedirs(out_dir, exist_ok=True)

        # Save latest SAE state each epoch
        latest = os.path.join(out_dir, "latest_sae.pth")
        torch.save(self.sae.state_dict(), latest)

        # Save after first epoch (epoch indexing starts at 0)
        if (self.current_epoch + 1) == 1:
            ep1 = os.path.join(out_dir, "sae_epoch1.pth")
            torch.save(self.sae.state_dict(), ep1)

        # Best-by-val
        val = self.trainer.callback_metrics.get("val_loss")
        try:
            cur = float(val.detach().cpu()) if val is not None else float("inf")
        except Exception:
            cur = float("inf")
        if cur < self.best_val:
            self.best_val = cur
            best = os.path.join(out_dir, "best_sae.pth")
            torch.save(self.sae.state_dict(), best)

    def configure_optimizers(self):
        lr = float(self.hparams.get("learning_rate") or self.hparams.get("lr") or 1e-3)
        wd = float(self.hparams.get("weight_decay") or 1e-2)
        opt = optim.AdamW(self.sae.parameters(), lr=lr, weight_decay=wd)
        # OneCycleLR over full training
        sched = OneCycleLR(
            opt,
            max_lr=lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def _build_loaders(cfg: Dict[str, Any]):
    # Build transforms using encoder-aware normalization
    tf_cfg = {
        "encoder_type": cfg["encoder_type"],
        "image_size": int(cfg.get("image_size", 256)),
        "freeze_encoder": True,
        # Optional augs respected if present
        **{k: v for k, v in cfg.items() if k in {
            "hflip", "vflip", "rot90s", "affine", "rrc", "rrc_scale",
            "color_jitter", "color_jitter_params", "color_jitter_p",
            "occlude", "occ_patch_size_x", "occ_patch_size_y", "occ_patch_vary_width",
            "occ_patch_min_size", "occ_patch_max_size", "occ_use_batch"
        }}
    }
    tf = build_transforms(tf_cfg)

    bs = int(cfg.get("batch_size", 32))
    nw = int(cfg.get("num_workers", 4))

    # Resolve dataset sources; prefer single_csv_path, then split CSVs, else directory + samples
    if cfg.get("single_csv_path"):
        train_ds = get_dataset_single_file(
            csv_path=cfg["single_csv_path"],
            data_dir=cfg.get("data_dir"),
            transforms=tf["train"],
            bins=int(cfg.get("bins", 1)),
            only_inputs=True,
            tile_subdir=cfg.get("tile_subdir"),
            split="train",
            split_col_name=cfg.get("split_col_name", "split"),
        )
        val_ds = get_dataset_single_file(
            csv_path=cfg["single_csv_path"],
            data_dir=cfg.get("data_dir"),
            transforms=tf["eval"],
            bins=int(cfg.get("bins", 1)),
            only_inputs=True,
            tile_subdir=cfg.get("tile_subdir"),
            split="val",
            split_col_name=cfg.get("split_col_name", "split"),
        )
    elif cfg.get("train_csv_path") and cfg.get("val_csv_path"):
        train_ds = get_dataset_single_file(
            csv_path=cfg["train_csv_path"],
            data_dir=cfg.get("data_dir"),
            transforms=tf["train"],
            bins=int(cfg.get("bins", 1)),
            only_inputs=True,
            tile_subdir=cfg.get("tile_subdir"),
        )
        val_ds = get_dataset_single_file(
            csv_path=cfg["val_csv_path"],
            data_dir=cfg.get("data_dir"),
            transforms=tf["eval"],
            bins=int(cfg.get("bins", 1)),
            only_inputs=True,
            tile_subdir=cfg.get("tile_subdir"),
        )
    else:
        # Directory + patient samples
        train_ds = get_dataset(
            data_dir=cfg["data_dir"],
            genes=None,  # let dataset infer; we ignore labels anyway
            transforms=tf["train"],
            samples=cfg.get("train_samples"),
            bins=int(cfg.get("bins", 1)),
            only_inputs=True,
            gene_data_filename=cfg.get("gene_data_filename", "gene_data.csv"),
            meta_data_dir=cfg.get("meta_data_dir", "/meta_data/"),
        )
        val_ds = get_dataset(
            data_dir=cfg["data_dir"],
            genes=None,
            transforms=tf["eval"],
            samples=cfg.get("val_samples"),
            bins=int(cfg.get("bins", 1)),
            only_inputs=True,
            gene_data_filename=cfg.get("gene_data_filename", "gene_data.csv"),
            meta_data_dir=cfg.get("meta_data_dir", "/meta_data/"),
        )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=torch.cuda.is_available()
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader


def _sanitize(s: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in str(s))


def _select_encoder(cfg: Dict[str, Any], override_name: Optional[str], index_1based: Optional[int]) -> str:
    # Accept either a single encoder_type or a list under encoder_type/encoders
    cand = cfg.get("encoder_type")
    enc_list = None
    if isinstance(cand, list):
        enc_list = cand
    elif isinstance(cfg.get("encoders"), list):
        enc_list = cfg["encoders"]

    if enc_list:
        if override_name:
            return override_name
        idx = int(index_1based or cfg.get("encoder_index", 1))
        idx = max(1, min(idx, len(enc_list)))
        return enc_list[idx - 1]
    else:
        return str(cand)


def _write_config(out_dir: str, cfg: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def parse_args():
    p = argparse.ArgumentParser(description="Train a Sparse Autoencoder on top of a frozen encoder.")
    p.add_argument("--config", "-c", required=True, type=str, help="Path to YAML config")
    p.add_argument("--encoder", type=str, default=None, help="Override encoder_type (if config lists multiple)")
    p.add_argument("--encoder_index", type=int, default=None, help="1-based index into encoder list in config")
    return p.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    raw = load_yaml(args.config)
    # Flatten out wandb sweep style configs if present
    cfg = {k: v for k, v in raw.items() if k not in ("parameters", "metric", "method")}

    # Resolve encoder selection
    enc = _select_encoder(cfg, args.encoder, args.encoder_index)
    if not enc:
        raise RuntimeError("No encoder_type resolved from config/args.")
    cfg = dict(cfg)
    cfg["encoder_type"] = enc

    # Resolve output directory (per-encoder subdir)
    # Place SAE outputs in a top-level 'sae' folder next to 'script' and 'encoders'
    repo_root = Path(__file__).resolve().parents[2]
    base_out = str(repo_root / "sae")
    out_dir = os.path.join(base_out, _sanitize(enc))
    cfg["out_dir"] = out_dir
    _write_config(out_dir, cfg)

    # Build loaders and module
    train_loader, val_loader = _build_loaders(cfg)
    # SAE hyperparams (top-level or nested under 'sae')
    trainer = L.Trainer(
        max_epochs=int(cfg.get("epochs") or cfg.get("sae", {}).get("epochs", 1)),
        precision=cfg.get("precision", "16-mixed" if torch.cuda.is_available() else 32),
        log_every_n_steps=int(cfg.get("log_every_n_steps", 50)),
        enable_progress_bar=True,
        gradient_clip_val=float(cfg.get("grad_clip", 0.0)),
        deterministic=bool(cfg.get("deterministic", False)),
    )

    # Learning settings for SAE
    sae_train_cfg = {
        "encoder_type": enc,
        "out_dir": out_dir,
        "learning_rate": float(cfg.get("learning_rate") or cfg.get("sae", {}).get("learning_rate", 1e-3)),
        "weight_decay": float(cfg.get("weight_decay") or cfg.get("sae", {}).get("weight_decay", 1e-2)),
        # SAE hyperparams
        "d_hidden": int(cfg.get("d_hidden") or cfg.get("sae", {}).get("d_hidden", 1024)),
        "k": int(cfg.get("k") or cfg.get("sae", {}).get("k", 64)),
        "allow_negative_topk": bool(cfg.get("allow_negative_topk") or cfg.get("sae", {}).get("allow_negative_topk", False)),
    }

    module = SAEOnEncoder(sae_train_cfg)
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
