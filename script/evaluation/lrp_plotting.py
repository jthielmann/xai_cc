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
