import torch
import torch.nn as nn
import torchvision.models as models
from script.configs.config_factory import get_dataset_cfg
import timm
import os
from huggingface_hub import login

def get_encoder(encoder_type: str) -> nn.Module:
    if encoder_type == "dino":
        return torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    if encoder_type == "resnet50random":
        return models.resnet50(encoder=False)
    if encoder_type == "resnet50imagenet":
        return models.resnet50(weights="IMAGENET1K_V2")
    if encoder_type == "unimodel":
        return load_uni_model()
    raise ValueError(f"Unknown encoder {encoder_type}")


def build_model(**kwargs):
    return timm.create_model(**kwargs)

def load_uni_model():
    model_file = "UNI2-h_state.pt"            # local cache of the weights
    # timm kwargs for the UNI2-h backbone
    timm_kwargs = {
        'model_name': 'vit_giant_patch14_224',
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }

    print("loading model from local file")
    model = build_model(**timm_kwargs)
    state_dict = torch.load(model_file, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def get_encoder_out_dim(encoder_type: str) -> int:
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
