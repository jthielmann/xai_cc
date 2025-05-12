import torch
import torch.nn as nn
import torchvision.models as models
from script.configs.config_factory import get_dataset_cfg

def get_encoder(encoder_type: str) -> nn.Module:
    if encoder_type == "dino":
        return torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    if encoder_type == "resnet50random":
        return models.resnet50(pretrained=False)
    if encoder_type == "resnet50imagenet":
        return models.resnet50(weights="IMAGENET1K_V2")
    raise ValueError(f"Unknown encoder {encoder_type}")

def get_pretrained_out_dim(encoder_type: str) -> int:
    return 2048 if encoder_type == "dino" else 1000


class WMSE(nn.Module):
    def __init__(self, w): super().__init__(); self.w = w

    def forward(self, x, y):
        loss = (x - y).pow(2) * self.w
        return loss.mean()

def get_loss_fn(kind: str, dataset: str) -> nn.Module:
    if kind == "MSE":
        return nn.MSELoss()
    if kind == "Weighted MSE":
        dataset = get_dataset_cfg(name=dataset, debug=False)
        return WMSE(dataset["weights"])
    raise ValueError(f"Unknown loss {kind}")
