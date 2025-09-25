import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import v2

from script.configs.config_factory import get_dataset_cfg
import timm
from typing import Tuple
from torchvision import transforms


def get_encoder(encoder_type: str) -> nn.Module:
    t = encoder_type.lower() # keep encoder_type var for logging on error later
    if t == "dino": return torch.hub.load('facebookresearch/dino:main','dino_resnet50')
    if t.startswith("dinov3"):
        import os
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "encoders"))
        return torch.hub.load(repo, t, source="local", weights=os.path.join(repo, f"{t}.pth"))
    if t == "resnet50random": return models.resnet50(weights=False)
    if t == "resnet50imagenet": return models.resnet50(weights="IMAGENET1K_V2")
    if t == "unimodel": return load_uni_model()
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


def make_transform_imagenet(resize_size: int = 224):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


def get_encoder_transforms(encoder_type: str):
    t = (encoder_type or "").lower()
    # Map encoder types → normalization
    if t == "dino" or t.startswith("resnet"):
        # I could not find mean std on the official dinov1 github, however they use imagenet
        transform = make_transform_imagenet()
    elif t.startswith("dinov3"):
        # DINOv3 uses standard ImageNet normalization in the official code: https://github.com/facebookresearch/dinov3
        transform = make_transform_imagenet()
    elif t in {"unimodel", "uni2", "uni2-h", "uni2h", "uni"}:
        # from the hf page: https://huggingface.co/MahmoodLab/UNI2-h
        transform = transforms.Compose(
            [
                v2.Resize(224),
                v2.ToTensor(),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    else:
        raise RuntimeError(f"Cannot deduct mean std for {encoder_type}")

    return transform

