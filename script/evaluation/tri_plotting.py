import numpy as np
import matplotlib.pyplot as plt
import logging
log = logging.getLogger(__name__)
import os
import wandb
import torch
import pandas as pd

from script.evaluation.relevance import get_coords_from_name
from script.data_processing.data_loader import get_patient_loader, get_dataset
from script.data_processing.image_transforms import get_transforms

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

    os.makedirs(out_path, exist_ok=True)
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
        # Also log a per-case table row with summary metrics
        mae = float(np.mean(np.abs(y_pred - y_label)))
        rmse = float(np.sqrt(np.mean((y_pred - y_label) ** 2)))
        n = int(len(y_label))
        table = wandb.Table(columns=["patient", "gene", "n_tiles", "mae", "rmse", "triptych"])
        table.add_data(str(patient), str(gene), n, mae, rmse, wandb.Image(fig))
        wandb_run.log({"diff/table": table})
        for f in singles.values():
            plt.close(f)

    plt.close(fig)
    log.info("Saved spatial plot: %s", out_file)


def plot_triptych_from_model(model, cfg: dict, patient: str, gene: str, out_path: str, *, max_items: int = None, is_online=False, wandb_run=None):
    data_dir = cfg.get("data_dir")
    if not data_dir:
        raise ValueError("Missing 'data_dir' in config; cannot build spatial dataset for triptych.")
    meta_data_dir = cfg.get("meta_data_dir", "/meta_data/")
    gene_data_filename = cfg.get("gene_data_filename", "gene_data.csv")

    # Build a standard dataset restricted to the selected patient and single gene
    # Coordinates (x,y) are expected inside the gene CSV for COAD and similar datasets.
    eval_tf = get_transforms(cfg.get("model_config", {}), split="eval")
    ds = get_dataset(
        data_dir=data_dir,
        genes=[gene],
        transforms=eval_tf,
        samples=[patient],
        max_len=max_items if (max_items is not None and max_items > 0) else None,
        only_inputs=False,
        meta_data_dir=meta_data_dir,
        gene_data_filename=gene_data_filename,
        return_floats=True,
    )
    if len(ds) == 0:
        raise ValueError(f"No tiles found for patient={patient} under {data_dir}")

    # Map gene name to model output index (assume present; fail fast if not)
    gene_idx = int(model.gene_to_idx[gene])

    # Run inference to collect predictions in dataset order
    device = next(model.parameters()).device
    model.eval()
    preds = []
    labels = []
    xs = []
    ys = []
    with torch.no_grad():
        limit = min(len(ds), int(max_items)) if (max_items is not None and max_items > 0) else len(ds)
        for i in range(limit):
            item = ds[i]
            # ds returns (img, target) in return_floats mode for standard datasets
            if isinstance(item, (list, tuple)):
                img, target = item[:2]
            else:
                img, target = item, None
            # Coordinates from the underlying DataFrame (populated by get_base_dataset)
            row = ds.df.iloc[i]
            if "x" not in row or "y" not in row:
                raise KeyError("Columns 'x' and 'y' not found in dataset; ensure your gene CSV contains them.")
            x_val = float(row["x"]) ; y_val = float(row["y"])
            img_t = img.unsqueeze(0).to(device) if hasattr(img, 'unsqueeze') else torch.from_numpy(img).unsqueeze(0).to(device)
            out = model(img_t)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out_vec = out[0].detach().cpu().flatten().numpy()
            if gene_idx >= len(out_vec):
                raise ValueError(
                    f"Model output size ({len(out_vec)}) smaller than index for gene {gene} ({gene_idx})."
                )
            preds.append(float(out_vec[gene_idx]))
            if target is not None:
                if isinstance(target, np.ndarray):
                    labels.append(float(target.reshape(-1)[0]))
                else:
                    labels.append(float(np.array(target).reshape(-1)[0]))
            else:
                # Fallback: read label from DataFrame
                labels.append(float(row[gene]))
            xs.append(float(x_val))
            ys.append(float(y_val))

    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    y_label = np.asarray(labels, dtype=np.float32)
    y_pred = np.asarray(preds, dtype=np.float32)

    plot_triptych(x, y, y_label, y_pred, patient, gene, out_path, is_online=is_online, wandb_run=wandb_run)
