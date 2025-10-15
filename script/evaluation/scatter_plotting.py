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

    # Load validation dataset with eval transforms, including targets
    eval_tf = get_transforms(config, split="eval")
    genes = config.get("genes")

    ds = get_dataset_from_config(
        dataset_name=config["dataset"],
        genes=genes,
        split="val",
        debug=bool(config.get("debug", False)),
        transforms=eval_tf,
        samples=None,
        only_inputs=False,
        max_len=100 if config["debug"] else None
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
    gene_names = list(genes) if genes is not None else [str(i) for i in range(y_hat.shape[1])]
    for gi, gene in enumerate(gene_names):
        yi = y_hat[:, gi]
        ti = y_true[:, gi]
        loss_g = float(torch.mean((yi - ti) ** 2))
        r_val = float(per_gene_r[gi]) if gi < len(per_gene_r) else float("nan")

        title_parts = ["val", str(gene)]
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
        try:
            fig.savefig(os.path.join(out_dir, f"scatter_{gene}.png"), dpi=150, bbox_inches="tight")
        except Exception:
            pass

        # Log to W&B if available
        if wandb_run is not None:
            try:
                wandb_run.log({f"scatter/{gene}": wandb.Image(fig)})
            except Exception:
                pass

        plt.close(fig)
