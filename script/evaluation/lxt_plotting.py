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

    # Build a dataset
    eval_tf = get_transforms(config["model_config"], split="eval")
    debug = bool(config.get("debug", False))

    # Prefer explicit list of tile paths if provided
    tilepaths = None
    for key in ("lxt_tilepaths", "lxt_tile_paths", "tilepaths", "tile_paths"):
        if key in config and config[key]:
            tilepaths = list(config[key])  # type: ignore[assignment]
            break

    if tilepaths is not None:
        # Normalize and filter non-existing paths with a gentle warning
        normalized: List[str] = []
        missing: List[str] = []
        for p in tilepaths:
            sp = os.path.expanduser(str(p))
            if os.path.exists(sp):
                normalized.append(sp)
            else:
                missing.append(sp)
        if missing and run is not None:
            run.log({"lxt/warnings": f"Skipping {len(missing)} missing tiles"}, commit=False)
        if not normalized:
            raise ValueError("No valid tile paths found in 'lxt_tilepaths'.")

        df = pd.DataFrame({"tile": normalized})
        ds = STDataset(df, image_transforms=eval_tf, inputs_only=True, genes=[])
        loader = DataLoader(ds, batch_size=1, shuffle=False)
    else:
        # Fall back to a small sample from the configured validation split
        ds = get_dataset_from_config(
            dataset_name=config["model_config"]["dataset"],
            genes=None,
            split="val",
            debug=debug,
            transforms=eval_tf,
            samples=None,
            only_inputs=True,
            meta_data_dir=config["model_config"].get("meta_data_dir", "/meta_data/"),
            gene_data_filename=config["model_config"].get("gene_data_filename", "gene_data.csv"),
        )
        n = min(10, len(ds))
        loader = DataLoader(Subset(ds, list(range(n))), batch_size=1, shuffle=False)

    device = next(model.parameters()).device
    model.eval()

    conv_list, lin_list = _resolve_gamma_config(config)

    any_logged = False
    last_img = None
    # Evaluate all gamma pairs; one grid per pair for clarity
    for conv_gamma, lin_gamma in itertools.product(conv_list, lin_list):
        heatmaps = []
        for batch in loader:
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
            denom = torch.clamp(hm.abs().amax(dim=(1, 2), keepdim=True), min=1e-12)
            hm = hm / denom
            heatmaps.append(hm[0].detach().cpu())

        if not heatmaps:
            continue
        print('heatmaps.shape:', np.array(heatmaps).shape)
        grid_img = imgify(heatmaps, vmin=-1, vmax=1)
        last_img = grid_img
        if run is not None:
            run.log({f"lxt/heatmaps[conv={conv_gamma},lin={lin_gamma}]": wandb.Image(grid_img)}, commit=True)
            any_logged = True

    if last_img is None:
        return None
    return last_img
