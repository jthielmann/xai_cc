"""Utility helpers for :mod:`lit_model`.

The goal is to keep :mod:`lit_model` focused on the Lightning module logic by
centralising the smaller utilities here.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torchmetrics.functional import pearson_corrcoef
from contextlib import nullcontext


SCATTER_KEY_ABBREVIATIONS = {
    "learning_rate": "lr",
    "batch_size": "bs",
    "dropout_rate": "dr",
    "loss_fn_switch": "loss",
    "encoder_type": "encdr",
    "middle_layer_features": "mfeatures",
    "gene_data_filename": "file",
    "freeze_encoder": "f_encdr",
    "one_linear_out_layer": "1linLr",
    "use_leaky_relu": "lkReLu",
    "use_early_stopping": "eStop",
}

RESULTS_ROW_CONFIG_KEYS = [
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


def bytes2gb(x: int) -> float:
    return x / (1024 ** 3)


@torch.no_grad()
def estimate_per_sample_activations(model: torch.nn.Module, sample: torch.Tensor) -> float:
    """Estimate the number of bytes of activations kept per sample."""
    sizes: list[int] = []
    handles = []

    def hook(_m, _inp, out):
        def num_bytes(t):
            if not torch.is_tensor(t):
                return 0
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
    _ = model(sample)
    for h in handles:
        h.remove()

    total_bytes = sum(sizes)
    return total_bytes / sample.shape[0]


def measure_peak_train_step(model, batch, criterion, optimizer, amp: bool = True, device: str = "cuda") -> int:
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

    return torch.cuda.max_memory_allocated(device)


def compute_per_gene_pearson(y_hat: torch.Tensor, y_true: torch.Tensor) -> list[float]:
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




def sanitize_for_json(value: Any) -> Any:
    try:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if 'torch' in globals():
            import torch as _torch  # ensure name exists locally
            if isinstance(value, _torch.Tensor):
                return value.detach().cpu().tolist()
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def build_config_columns(config: Dict[str, Any], tuned_lrs: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg_cols = {f"cfg_{k}": sanitize_for_json(v) for k, v in config.items()}
    if tuned_lrs is not None:
        cfg_cols["cfg_tuned_lr"] = sanitize_for_json(tuned_lrs)
    try:
        full_cfg = {k: sanitize_for_json(v) for k, v in config.items()}
        cfg_cols["hp_json"] = json.dumps(full_cfg, ensure_ascii=False)
    except Exception:
        cfg_cols["hp_json"] = json.dumps({k: str(v) for k, v in config.items()}, ensure_ascii=False)
    return cfg_cols


def append_row_with_schema(csv_path: str, row: Dict[str, Any]) -> None:
    df_new = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        all_cols = list(dict.fromkeys(list(df_old.columns) + list(df_new.columns)))
        df_old = df_old.reindex(columns=all_cols)
        df_new = df_new.reindex(columns=all_cols)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False)


def select_per_gene_values(
    best_pearson_per_gene: Sequence[float],
    best_r: Sequence[float],
    last_r: Sequence[float],
    expected_len: int,
) -> list[float]:
    for values in (best_pearson_per_gene, best_r, last_r):
        if values:
            if len(values) != expected_len:
                raise RuntimeError(f"genes ({expected_len}) vs r ({len(values)}) length mismatch")
            return [float(v) for v in values]
    if expected_len:
        raise RuntimeError(f"genes ({expected_len}) vs r (0) length mismatch")
    return []


def build_results_row(
    config: Dict[str, Any],
    genes: Sequence[str],
    metrics: Dict[str, Any],
    per_gene_values: Sequence[float],
    tuned_lrs: Dict[str, Any] = None,
) -> Dict[str, Any]:
    row = {
        "best_epoch": metrics["best_epoch"],
        "val_score": metrics["best_loss"],
        "pearson_mean": metrics["pearson_mean"],
        "out_path": config["out_path"],
        "model_path": metrics["best_model_path"],
        "wandb_url": metrics.get("wandb_url", ""),
    }
    if metrics.get("best_pearson_epoch") is not None:
        row["best_pearson_epoch"] = metrics["best_pearson_epoch"]

    seen: Dict[str, int] = {}
    for gene, value in zip(genes, per_gene_values):
        base_key = f"pearson_{gene}"
        index = seen.get(gene, 0)
        key = f"{base_key}__{index}" if index else base_key
        seen[gene] = index + 1
        row[key] = float(value)

    for key in RESULTS_ROW_CONFIG_KEYS:
        if key in config:
            row[key] = config[key]

    row.update(build_config_columns(config, tuned_lrs))
    return row


def build_scatter_appendix(config: Dict[str, Any], sweep_keys: Iterable[str]) -> str:
    if not sweep_keys:
        return ""
    parts = []
    for key in sweep_keys:
        parts.append(f"{SCATTER_KEY_ABBREVIATIONS.get(key, key)}={config.get(key)}")
    return " | ".join(parts)
