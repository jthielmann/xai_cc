import numpy as np
import matplotlib.pyplot as plt
import logging
log = logging.getLogger(__name__)
import os
import wandb
import pandas as pd

from script.evaluation.relevance import get_coords_from_name
from script.data_processing.data_loader import get_patient_loader

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



def plot_triptych(x, y, y_label, y_pred, patient, gene, out_path, is_online=False, wandb_run=None):
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


def plot_triptych_from_merge(data_dir, patient, gene, out_path, is_online=False, wandb_run=None):
    base = os.path.join(data_dir, patient, "meta_data")
    df = pd.read_csv(os.path.join(base, "merge.csv"))
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    y_label = df["labels"].to_numpy()
    y_pred = df["output"].to_numpy()
    plot_triptych(x, y, y_label, y_pred, patient, gene, out_path, is_online=is_online, wandb_run=wandb_run)
