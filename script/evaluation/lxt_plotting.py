import itertools
from typing import Optional, Tuple, List
import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Subset

from torchvision.models import vision_transformer

from zennit.image import imgify
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules

from lxt.efficient import monkey_patch, monkey_patch_zennit

from script.data_processing.image_transforms import get_transforms
from script.data_processing.data_loader import get_dataset_from_config, STDataset

import wandb


def _resolve_gamma_config(config) -> Tuple[List[float], List[float]]:
    """Resolve gamma configuration from config without defaults.

    Supports either single values via `lxt_conv_gamma` and `lxt_lin_gamma`,
    or lists via `lxt_conv_gamma_list` and `lxt_lin_gamma_list`.
    Raises if neither form is provided.
    """
    if ("lxt_conv_gamma" in config) or ("lxt_lin_gamma" in config):
        if ("lxt_conv_gamma" not in config) or ("lxt_lin_gamma" not in config):
            raise ValueError("Both 'lxt_conv_gamma' and 'lxt_lin_gamma' must be provided.")
        try:
            cg = float(config["lxt_conv_gamma"])  # type: ignore[arg-type]
            lg = float(config["lxt_lin_gamma"])   # type: ignore[arg-type]
        except Exception as e:
            raise ValueError("'lxt_conv_gamma' and 'lxt_lin_gamma' must be numeric.") from e
        return [cg], [lg]

    if ("lxt_conv_gamma_list" in config) and ("lxt_lin_gamma_list" in config):
        conv_list = list(config["lxt_conv_gamma_list"])  # type: ignore
        lin_list = list(config["lxt_lin_gamma_list"])    # type: ignore
        if not conv_list or not lin_list:
            raise ValueError("Gamma lists must be non-empty.")
        try:
            conv_list = [float(v) for v in conv_list]
            lin_list = [float(v) for v in lin_list]
        except Exception as e:
            raise ValueError("Values in gamma lists must be numeric.") from e
        return conv_list, lin_list

    raise ValueError(
        "No gamma configuration provided. Specify either ('lxt_conv_gamma' & 'lxt_lin_gamma') "
        "or ('lxt_conv_gamma_list' & 'lxt_lin_gamma_list')."
    )


def plot_lxt(model, config, run: Optional["wandb.sdk.wandb_run.Run"] = None):
    # Ensure ViT and zennit are patched for LXT
    monkey_patch(vision_transformer, verbose=True)
    monkey_patch_zennit(verbose=True)

    # Build a dataset from test split, honoring optional test_samples
    eval_tf = get_transforms(config["model_config"], split="eval")
    debug = bool(config.get("debug", False))

    # Directly resolve metadata CSV using eval override with model_config fallback
    cfg = config
    meta_dir = cfg.get("meta_data_dir") or cfg["model_config"].get("meta_data_dir", "/meta_data/")
    gene_csv = cfg.get("gene_data_filename") or cfg.get("model_config", {}).get("gene_data_filename", "gene_data.csv")
    ds = get_dataset_from_config(
        dataset_name=config["model_config"]["dataset"],
        genes=None,
        split="test",
        debug=debug,
        transforms=eval_tf,
        samples=config.get("test_samples"),
        max_len=config.get("max_len") if debug else None,
        only_inputs=False,
        meta_data_dir=meta_dir,
        gene_data_filename=gene_csv,
    )
    # Limit number of items evaluated to speed up debug runs
    n_items = min(int(config.get("lxt_max_items", len(ds))), len(ds))
    loader = DataLoader(Subset(ds, list(range(n_items))), batch_size=1, shuffle=False)

    device = next(model.parameters()).device
    model.eval()

    conv_list, lin_list = _resolve_gamma_config(config)

    any_logged = False
    last_img = None
    out_dir = config.get("out_path")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # Evaluate all gamma pairs; one grid per pair for clarity
    table_rows = []
    for conv_gamma, lin_gamma in itertools.product(conv_list, lin_list):
        for i, batch in enumerate(loader):
            x = batch if not isinstance(batch, (tuple, list)) else batch[0]
            x = x.to(device)
            x = x.detach().requires_grad_(True)
            model.zero_grad(set_to_none=True)

            comp = LayerMapComposite([
                (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
                (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
            ])
            comp.register(model)

            y = model(x)
            if y.dim() == 1:
                y = y.unsqueeze(0)
            y[0, 0].backward()
            comp.remove()

            hm = (x * x.grad).sum(1)
            hm = torch.nan_to_num(hm, nan=0.0, posinf=1.0, neginf=-1.0)
            denom = torch.clamp(hm.abs().amax(dim=(1, 2), keepdim=True), min=1e-12)
            hm = hm / denom
            heatmap = hm[0].detach().cpu()
            grid_img = imgify(np.asarray(heatmap)[..., None], vmin=-1, vmax=1)
            last_img = grid_img
            if run is not None:
                run.log({f"lxt/heatmaps[conv={conv_gamma},lin={lin_gamma}]": wandb.Image(grid_img)}, commit=True)
                # Accumulate per-sample rows for a W&B table
                patient = None
                tile = None
                if isinstance(batch, (tuple, list)) and len(batch) >= 4:
                    patient, tile = batch[2], batch[3]
                    if isinstance(patient, (list, tuple)) and len(patient) > 0:
                        patient = patient[0]
                    if isinstance(tile, (list, tuple)) and len(tile) > 0:
                        tile = tile[0]
                tile_name = os.path.basename(tile) if isinstance(tile, str) else ""
                table_rows.append({
                    "idx": i,
                    "patient": patient or "",
                    "tile": tile_name,
                    "conv_gamma": float(conv_gamma),
                    "lin_gamma": float(lin_gamma),
                    "heatmap": wandb.Image(grid_img),
                })
                any_logged = True
            if out_dir:
                fn = f"lxt_conv={conv_gamma}_lin={lin_gamma}_{i:04d}.png"
                safe_fn = fn.replace(" ", "_")
                grid_img.save(os.path.join(out_dir, safe_fn))


    # Log a per-case W&B table at the end
    if run is not None and table_rows:
        cols = ["idx", "patient", "tile", "conv_gamma", "lin_gamma", "heatmap"]
        table = wandb.Table(columns=cols)
        for r in table_rows:
            table.add_data(r["idx"], r["patient"], r["tile"], r["conv_gamma"], r["lin_gamma"], r["heatmap"])
        run.log({"lxt/table": table})

    if last_img is None:
        return None
    return last_img
