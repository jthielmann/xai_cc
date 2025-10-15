import torch
from zennit.attribution import Gradient
from zennit.composites import EpsilonPlusFlat
import zennit.image as zimage
import wandb


def _imgify_rel(att):
    rel = att.sum(1).cpu()
    rel = rel / (abs(rel).max() + 1e-12)
    return zimage.imgify(rel, symmetric=True, cmap='coldnhot', vmin=-1, vmax=1)


def plot_lrp(model, data, run=None):
    model.eval()
    composite = EpsilonPlusFlat()
    for idx, sample in enumerate(data):
        with Gradient(model, composite) as attributor:
            _, grad = attributor(sample)
        img = _imgify_rel(grad)
        if run is not None:
            run.log({f"lrp/attribution[{idx}]": wandb.Image(img)})


def plot_lrp_custom(model, data, run=None):
    """Simple LRP-like visualization using vanilla input gradients.

    Treats the target as the sum over outputs and visualizes d(sum(outputs))/d(input).
    """
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
