import os
import torch
from zennit.attribution import Gradient
from zennit.composites import EpsilonPlusFlat
import zennit.image as zimage
import wandb
import numpy as np
from PIL import Image
from torchvision.utils import make_grid

def _imgify_rel(att):
    rel = att.sum(1).cpu()
    rel = torch.nan_to_num(rel, nan=0.0, posinf=1.0, neginf=-1.0)
    rel = rel / (abs(rel).nan_to_num(nan=0.0).max() + 1e-12)
    return zimage.imgify(rel, symmetric=True, cmap='coldnhot', vmin=-1, vmax=1)


def plot_lrp(model, data, run=None, save_dir: str = None):
    """Compute LRP attributions and either log to W&B and/or save locally.

    - If a W&B run is provided, log images under "lrp/".
    - If save_dir is provided, write PNGs there.
    - At least one of (run, save_dir) must be provided.
    """
    if run is None and not save_dir:
        raise RuntimeError(
            "No output target for LRP: provide a W&B run and/or save_dir."
        )
    model.eval()
    device = next(model.parameters()).device
    composite = EpsilonPlusFlat()
    genes = model.genes
    if genes is None:
        raise AttributeError("model has no genes attribute for per-gene LRP")
    table_rows = []  # accumulate per-sample rows for a W&B table

    for idx, sample in enumerate(data):
        # Support datasets returning (img, y, patient, tile)
        x = sample[0] if isinstance(sample, (tuple, list)) else sample
        patient = (
            sample[2]
            if isinstance(sample, (tuple, list)) and len(sample) >= 3
            else None
        )
        tile = (
            sample[3]
            if isinstance(sample, (tuple, list)) and len(sample) >= 4
            else None
        )
        # De-batch patient/tile if they were collated into length-1 lists by DataLoader
        if isinstance(patient, (list, tuple)) and len(patient) > 0:
            patient = patient[0]
        if isinstance(tile, (list, tuple)) and len(tile) > 0:
            tile = tile[0]
        x_dev = x.to(device)
        if x_dev.dim() == 3:
            x_dev = x_dev.unsqueeze(0)
        for gene_idx, gene in enumerate(genes):
            def _sel(out, idx=gene_idx):
                return out[:, idx].sum()
            with Gradient(model, composite, attr_output_fn=_sel) as attributor:
                out, grad = attributor(x_dev)

            # Build a side-by-side panel: input (left) + attribution (right)
            img_attr = _imgify_rel(grad).convert("RGB")
            grid = make_grid(x_dev, nrow=1, normalize=True).cpu()
            img_in = Image.fromarray(
                (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
            panel = Image.new("RGB", (img_in.width + img_attr.width, img_in.height))
            panel.paste(img_in, (0, 0))
            panel.paste(img_attr, (img_in.width, 0))

            # Collect simple selection metrics for a per-case W&B table
            rel = grad.sum(1).detach()[0]
            abs_sum = rel.abs().sum().item()
            pos_frac = (rel > 0).float().mean().item()
            max_abs = rel.abs().max().item()
            thr = max_abs * 0.1
            coverage = (rel.abs() > thr).float().mean().item() if max_abs > 0 else 0.0
            pred_mean = float(out.detach().mean().item())
            tile_name = os.path.basename(tile) if isinstance(tile, str) else ""
            table_rows.append({
                "idx": idx,
                "gene": gene,
                "patient": patient or "",
                "tile": tile_name,
                "pred_mean": pred_mean,
                "rel_abs_sum": abs_sum,
                "pos_frac": pos_frac,
                "coverage@10%": coverage,
                "panel": wandb.Image(panel),
            })

            if run is not None:
                caption = None
                if patient or tile:
                    t = os.path.basename(tile) if isinstance(tile, str) else tile
                    caption = f"patient={patient}  tile={t}" if patient else f"tile={t}"
                key = f"lrp/attribution_{idx}_{gene}"
                run.log({key: wandb.Image(panel, caption=caption)}, commit=True)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fn = os.path.join(save_dir, f"lrp_{idx:04d}_{gene}.png")
                panel.save(fn)

    # Log a per-case W&B table at the end (if W&B run is provided)
    if run is not None and len(table_rows) > 0:
        cols = [
            "idx",
            "gene",
            "patient",
            "tile",
            "pred_mean",
            "rel_abs_sum",
            "pos_frac",
            "coverage@10%",
            "panel",
        ]
        table = wandb.Table(columns=cols)
        for r in table_rows:
            table.add_data(
                r["idx"],
                r["gene"],
                r["patient"],
                r["tile"],
                r["pred_mean"],
                r["rel_abs_sum"],
                r["pos_frac"],
                r["coverage@10%"],
                r["panel"],
            )
        run.log({"lrp/table": table})


def plot_lrp_custom(model, data, run=None, save_dir: str = None):
    model.eval()
    device = next(model.parameters()).device
    table_rows = []
    for idx, sample in enumerate(data):
        x = sample[0] if isinstance(sample, (tuple, list)) else sample
        patient = (
            sample[2]
            if isinstance(sample, (tuple, list)) and len(sample) >= 3
            else None
        )
        tile = (
            sample[3]
            if isinstance(sample, (tuple, list)) and len(sample) >= 4
            else None
        )
        if isinstance(patient, (list, tuple)) and len(patient) > 0:
            patient = patient[0]
        if isinstance(tile, (list, tuple)) and len(tile) > 0:
            tile = tile[0]
        x = x.to(device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.detach().requires_grad_(True)
        y = model(x)
        y.sum().backward()
        grad = x.grad
        # Build a side-by-side panel: input (left) + attribution (right)
        img_attr = _imgify_rel(grad).convert("RGB")
        grid = make_grid(x, nrow=1, normalize=True).detach().cpu()
        img_in = Image.fromarray(
            (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )
        panel = Image.new("RGB", (img_in.width + img_attr.width, img_in.height))
        panel.paste(img_in, (0, 0))
        panel.paste(img_attr, (img_in.width, 0))
        # Table metrics
        rel = grad.sum(1).detach()[0]
        abs_sum = rel.abs().sum().item()
        pos_frac = (rel > 0).float().mean().item()
        max_abs = rel.abs().max().item()
        thr = max_abs * 0.1
        coverage = (rel.abs() > thr).float().mean().item() if max_abs > 0 else 0.0
        pred_mean = float(y.detach().mean().item())
        tile_name = os.path.basename(tile) if isinstance(tile, str) else ""
        table_rows.append({
            "idx": idx,
            "patient": patient or "",
            "tile": tile_name,
            "pred_mean": pred_mean,
            "rel_abs_sum": abs_sum,
            "pos_frac": pos_frac,
            "coverage@10%": coverage,
            "panel": wandb.Image(panel),
        })
        if run is not None:
            run.log({f"lrp/attribution_custom[{idx}]": wandb.Image(panel)})
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fn = os.path.join(save_dir, f"lrp_custom_{idx:04d}.png")
            panel.save(fn)
    if run is not None and table_rows:
        cols = [
            "idx",
            "patient",
            "tile",
            "pred_mean",
            "rel_abs_sum",
            "pos_frac",
            "coverage@10%",
            "panel",
        ]
        table = wandb.Table(columns=cols)
        for r in table_rows:
            table.add_data(
                r["idx"],
                r["patient"],
                r["tile"],
                r["pred_mean"],
                r["rel_abs_sum"],
                r["pos_frac"],
                r["coverage@10%"],
                r["panel"],
            )
        run.log({"lrp/table_custom": table})
