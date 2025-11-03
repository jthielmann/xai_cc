import os
from typing import Optional, Iterable

import numpy as np
from matplotlib import cm, colors, pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import wandb

from script.data_processing.data_loader import get_dataset_from_config
from script.data_processing.image_transforms import get_transforms
from script.model.lit_model_helpers import compute_per_gene_pearson, build_scatter_appendix


def make_scatter_figure(yh: np.ndarray, yt: np.ndarray, loss: float, r: float, title: str):
    lo, hi = float(min(yh.min(), yt.min())), float(max(yh.max(), yt.max()))
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    fig, ax = plt.subplots()
    coords = np.column_stack((yh, yt))
    unique_coords, counts = np.unique(coords, axis=0, return_counts=True)
    cmap = cm.get_cmap("Blues")
    vmax = max(int(np.max(counts)), 2)
    norm = colors.Normalize(vmin=1, vmax=vmax)
    facecolors = cmap(norm(counts))
    ax.scatter(unique_coords[:, 0], unique_coords[:, 1], s=8, c=facecolors, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], linewidth=1)
    ax.text(0.02, 0.98, f"loss: {loss:.3f}\nr: {r:.3f}", transform=ax.transAxes, va="top", ha="left", fontsize="small")
    ax.set(title=title, xlabel="output", ylabel="target")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Samples per point", fontsize="small")
    cbar.ax.tick_params(labelsize="x-small")
    return fig


@torch.no_grad()
def plot_scatter(config, model, wandb_run=None):
    model.eval()

    # Resolve device from model parameters
    device = next(model.parameters()).device

    # Load test dataset with eval transforms, including targets
    # Prefer the trained model's config for encoder-dependent transforms
    base_cfg = config.get("model_config", config)
    eval_tf = get_transforms(base_cfg, split="eval")
    genes = config.get("genes")

    # Directly resolve metadata CSV using eval override with model_config fallback
    cfg = config
    meta_dir = cfg.get("meta_data_dir") or base_cfg.get("meta_data_dir", "/meta_data/")
    gene_csv = cfg.get("gene_data_filename") or cfg.get("model_config", {}).get("gene_data_filename", "gene_data.csv")
    ds = get_dataset_from_config(
        dataset_name=config["dataset"],
        genes=genes,
        split="test",
        debug=bool(config.get("debug", False)),
        transforms=eval_tf,
        samples=config.get("test_samples"),
        only_inputs=False,
        max_len=(int(config.get("max_len")) if bool(config.get("debug", False)) and config.get("max_len") is not None else None),
        meta_data_dir=meta_dir,
        gene_data_filename=gene_csv,
    )

    # Optional truncation for speed
    max_items = int(config.get("scatter_max_items", 0) or 0)
    if max_items > 0:
        max_items = min(max_items, len(ds))
        ds_eval = Subset(ds, list(range(max_items)))
    else:
        ds_eval = ds

    batch_size = int(config.get("scatter_batch_size", 32) or 32)
    loader = DataLoader(ds_eval, batch_size=batch_size, shuffle=False)

    preds = []
    targs = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1]
        else:
            # Dataset should return (input, target). If not, skip.
            continue
        x = x.to(device)
        y_hat = model(x)
        preds.append(y_hat.detach().cpu().float())
        targs.append(torch.as_tensor(y).detach().cpu().float())

    if not preds or not targs:
        return

    y_hat = torch.cat(preds, dim=0)
    y_true = torch.cat(targs, dim=0)

    # Compute per-gene Pearson r
    per_gene_r = compute_per_gene_pearson(y_hat, y_true)

    # Prepare title appendix (helpful in sweeps)
    sweep_cfg = getattr(getattr(wandb_run, "config", None), "_items", None) or getattr(wandb_run, "config", {})
    appendix = build_scatter_appendix(config, sweep_cfg.get("sweep_parameter_names", [])) if sweep_cfg else ""

    os.makedirs(config.get("out_path", "."), exist_ok=True)

    # One figure per gene
    table_rows = []
    gene_names = list(genes) if genes is not None else [str(i) for i in range(y_hat.shape[1])]
    for gi, gene in enumerate(gene_names):
        yi = y_hat[:, gi]
        ti = y_true[:, gi]
        loss_g = float(torch.mean((yi - ti) ** 2))
        r_val = float(per_gene_r[gi]) if gi < len(per_gene_r) else float("nan")

        title_parts = ["test", str(gene)]
        if appendix:
            title_parts.append(appendix)
        fig = make_scatter_figure(
            yi.cpu().numpy(),
            ti.cpu().numpy(),
            loss_g,
            r_val,
            " — ".join(title_parts),
        )

        # Save locally
        out_dir = config.get("out_path", ".")
        fig.savefig(os.path.join(out_dir, f"scatter_{gene}.png"), dpi=150, bbox_inches="tight")

        # Log to W&B if available
        if wandb_run is not None:
            img = wandb.Image(fig)
            wandb_run.log({f"scatter/{gene}": img})
            table_rows.append({
                "gene": str(gene),
                "loss": loss_g,
                "pearson_r": r_val,
                "figure": img,
            })

        plt.close(fig)

    # Per-case table at the end
    if wandb_run is not None and table_rows:
        cols = ["gene", "loss", "pearson_r", "figure"]
        table = wandb.Table(columns=cols)
        for r in table_rows:
            table.add_data(r["gene"], r["loss"], r["pearson_r"], r["figure"])
        wandb_run.log({"scatter/table": table})
