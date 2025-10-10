import torch
from zennit.attribution import Gradient
from zennit.composites import EpsilonPlusFlat
import zennit.image as zimage
import wandb


def _imgify_rel(att):
    rel = att.sum(1).cpu()
    rel = rel / (abs(rel).max() + 1e-12)
    return zimage.imgify(rel, symmetric=True, cmap='coldnhot', vmin=-1, vmax=1)


def plot_lrp(model, device=None, run=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    composite = EpsilonPlusFlat()
    with Gradient(model, composite) as attributor:
        _, grad = attributor(x)
    img = _imgify_rel(grad)
    if run is not None:
        run.log({"lrp/attribution": wandb.Image(img)})
