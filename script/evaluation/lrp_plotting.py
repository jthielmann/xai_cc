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


def plot_lrp(model, data, run=None, save_dir: str | None = None):
    """Compute LRP attributions and either log to W&B and/or save locally.

    - If a W&B run is provided, log images under "lrp/".
    - If save_dir is provided, write PNGs there.
    - At least one of (run, save_dir) must be provided.
    """
    if run is None and not save_dir:
        raise RuntimeError("No output target for LRP: provide a W&B run and/or save_dir.")
    model.eval()
    device = next(model.parameters()).device
    composite = EpsilonPlusFlat()

    for idx, sample in enumerate(data):
        with Gradient(model, composite) as attributor:
            _, grad = attributor(sample.to(device))

        # Build a side-by-side panel: input (left) + attribution (right)
        img_attr = _imgify_rel(grad).convert("RGB")
        grid = make_grid(sample, nrow=1, normalize=True).cpu()
        img_in = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        panel = Image.new("RGB", (img_in.width + img_attr.width, img_in.height))
        panel.paste(img_in, (0, 0))
        panel.paste(img_attr, (img_in.width, 0))

        if run is not None:
            run.log({f"lrp/attribution_{idx}": wandb.Image(panel)}, commit=True)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fn = os.path.join(save_dir, f"lrp_{idx:04d}.png")
            panel.save(fn)


def plot_lrp_custom(model, data, run=None, save_dir: str | None = None):
    model.eval()
    device = next(model.parameters()).device
    for idx, sample in enumerate(data):
        x = sample[0] if isinstance(sample, (tuple, list)) else sample
        x = x.to(device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.detach().requires_grad_(True)
        y = model(x)
        y.sum().backward()
        grad = x.grad
        img = _imgify_rel(grad)
        if run is not None:
            run.log({f"lrp/attribution_custom[{idx}]": wandb.Image(img)})
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fn = os.path.join(save_dir, f"lrp_custom_{idx:04d}.png")
            img.convert("RGB").save(fn)
