import os

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import v2

from script.configs.config_factory import get_dataset_cfg
import timm
from typing import Tuple

from typing import Iterable, Sequence, Callable, Optional

# resolver.py (or inline in your loader code)
from pathlib import Path
from typing import Iterable

def resolve_unique_model_file(
    encoder_type: str,
    encoders_dir: str | Path = "../encoders",
) -> Path:
    # Map: hub callable name -> checkpoint filename
    DINOv3_FILEMAP = {
        "dinov3_convnext_base" : "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
        "dinov3_convnext_large" : "dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth",
        "dinov3_convnext_tiny": "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
        "dinov3_vit7b16": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
        "dinov3_vith16plus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
        "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    }

    t = (encoder_type or "").strip().lower()

    if t in DINOv3_FILEMAP:
        path = encoders_dir + "/" + DINOv3_FILEMAP[t]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Resolved '{t}' to '{path.name}', but it does not exist in {encoders_dir}.")
        return Path(path)

    for key, value in DINOv3_FILEMAP.items():
        if key in encoder_type:
            return Path(encoders_dir, value)
    raise RuntimeError(f"{encoder_type} not in {DINOv3_FILEMAP}")




def get_encoder(encoder_type: str) -> nn.Module:
    t = encoder_type.lower() # keep encoder_type var for logging on error later
    if t == "dino": return torch.hub.load('facebookresearch/dino:main','dino_resnet50')
    if t.startswith("dinov3"):
        weights = resolve_unique_model_file(encoder_type)
        return torch.hub.load("../encoders/", encoder_type, source="local", weights=str(weights))
    if t == "resnet50random": return models.resnet50(weights=False)
    if t == "resnet50imagenet": return models.resnet50(weights="IMAGENET1K_V2")
    if t == "unimodel" or "uni2" or "uni": return load_uni_model()
    raise ValueError(f"Unknown encoder {encoder_type}")


def infer_encoder_out_dim(encoder: nn.Module,
                          input_size: Tuple[int,int,int]=(3,224,224),
                          device: torch.device=None) -> int:
    was_training = encoder.training
    encoder.eval()
    if device is None:
        device = next(encoder.parameters()).device
    dummy = torch.zeros(1, *input_size, device=device)
    with torch.no_grad():
        out = encoder(dummy)

    # If encoder outputs spatial maps, flatten them:
    if out.ndim > 2:
        out = torch.flatten(out, 1)

    # restore original mode
    if was_training:
        encoder.train()
    return out.size(1)


def build_model(**kwargs):
    return timm.create_model(**kwargs)


# similar to https://huggingface.co/MahmoodLab/UNI2-h
def load_uni_model():
    model_file = "../encoders/UNI2-h_state.pt"            # local cache of the weights
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


def _imagenet_parts(resize_size: int = 224):
    pre = [
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ]
    norm = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return pre, norm


def _build_pipeline(base_pre: Sequence[Callable], norm: Callable,
                    extra: Optional[Iterable[Callable]] = None,
                    place: str = "pre_norm") -> v2.Compose:
    extra = list(extra or [])
    if place == "pre_norm":
        steps = [*base_pre, *extra, norm]
    elif place == "post_norm":
        steps = [*base_pre, norm, *extra]
    else:
        raise ValueError(f"Unknown place='{place}' (use 'pre_norm' or 'post_norm').")
    return v2.Compose(steps)


def get_encoder_transforms(encoder_type: str,
                           resize_size: int = 224,
                           extra: Optional[Iterable[Callable]] = None,
                           place: str = "pre_norm") -> v2.Compose:
    """
    `extra`: optional iterable of v2 transforms to insert (default before Normalize).
    Typical occlusion/erasing goes BEFORE Normalize and AFTER ToDtype.
    """
    t = (encoder_type or "").lower()

    # All listed encoders here use ImageNet mean/std
    if t == "dino" or t.startswith("resnet") or t.startswith("dinov3") \
       or t.startswith("uni"):
        base_pre, norm = _imagenet_parts(resize_size)
    else:
        raise RuntimeError(f"Cannot deduct mean/std for {encoder_type}")

    return _build_pipeline(base_pre, norm, extra=extra, place=place)