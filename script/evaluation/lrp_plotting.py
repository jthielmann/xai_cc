import torch
from zennit.attribution import Gradient
from zennit.composites import EpsilonPlusFlat
import zennit.image as zimage
import wandb
import numpy as np

def _imgify_rel(att):
    rel = att.sum(1).cpu()
    rel = torch.nan_to_num(rel, nan=0.0, posinf=1.0, neginf=-1.0)
    rel = rel / (abs(rel).nan_to_num(nan=0.0).max() + 1e-12)
    return zimage.imgify(rel, symmetric=True, cmap='coldnhot', vmin=-1, vmax=1)


def plot_lrp(model, data, run):
    if run is None:
        raise RuntimeError("W&B run is None. Call wandb.init(...) and pass it in.")

    model.eval()
    composite = EpsilonPlusFlat()

    for idx, sample in enumerate(data):
        with Gradient(model, composite) as attributor:
            _, grad = attributor(sample)

        img = _imgify_rel(grad).convert("RGB")
        run.log({f"lrp/attribution_{idx}": wandb.Image(img)}, commit=True)


def plot_lrp_custom(model, data, run=None):
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
