import torch
import wandb
import torch.nn as nn
from crp.attribution import CondAttribution
from zennit.composites import EpsilonPlusFlat
from zennit.torchvision import VGGCanonizer, ResNetCanonizer
import zennit.image as zimage


def _get_layer_names(model, types):
    names = []
    for n, m in model.named_modules():
        for t in types:
            if isinstance(m, t):
                names.append(n)
                break
    return names


def _get_composite_and_layer(encoder):
    if type(encoder).__name__ == "VGG":
        composite = EpsilonPlusFlat(canonizers=[VGGCanonizer()])
        layer_type = nn.Linear
        layer_name = _get_layer_names(encoder, [nn.Linear])[-3]
    else:
        composite = EpsilonPlusFlat(canonizers=[ResNetCanonizer()])
        layer_type = type(encoder.layer1[0])
        layer_name = _get_layer_names(encoder, [layer_type])[-1]
    return composite, layer_name


def plot_crp(model, device=None, run=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    composite, layer_name = _get_composite_and_layer(model)
    x = torch.randn(1, 3, 224, 224, device=device).requires_grad_(True)
    attribution = CondAttribution(model)
    attr = attribution(x, [{"y": [0]}], composite, record_layer=[layer_name])
    rel = attr.relevances[layer_name]
    rel = rel.sum(1).cpu()
    rel = rel / (abs(rel).max() + 1e-12)
    img = zimage.imgify(rel, symmetric=True, cmap='coldnhot', vmin=-1, vmax=1)
    if run is not None:
        run.log({"crp/attribution": wandb.Image(img)})
