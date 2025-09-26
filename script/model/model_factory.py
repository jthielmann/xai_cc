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
    encoders_dir: str | Path,
    encoder_type: str,
    exts: Iterable[str] = (".pth", ".pt", ".ckpt", ".bin"),
    recursive: bool = False,
) -> Path:
    encoders_dir = Path(encoders_dir)
    key = encoder_type.strip()
    if not key:
        raise ValueError("encoder_type is empty.")
    direct = encoders_dir / key
    if direct.is_file():
        return direct

    exts_lower = {e.lower() for e in exts}
    files = encoders_dir.rglob("*") if recursive else encoders_dir.glob("*")
    key_lower = key.lower()

    candidates = [
        p for p in files
        if p.is_file() and key_lower in p.name.lower() and p.suffix.lower() in exts_lower
    ]

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        nearby = [p.name for p in encoders_dir.glob("*") if p.is_file()]
        hint = f"No file containing '{encoder_type}' with extensions {sorted(exts_lower)} in {encoders_dir}."
        if nearby:
            hint += f" Nearby files: {sorted(nearby)[:10]}"
        raise FileNotFoundError(hint)

    pretty = "\n  - " + "\n  - ".join(str(p) for p in sorted(candidates))
    raise ValueError(
        f"Ambiguous encoder match for '{encoder_type}' — multiple files contain that name:{pretty}\n"
        f"Please use a more specific key or pass an exact filename."
    )



def get_encoder(encoder_type: str) -> nn.Module:
    t = encoder_type.lower() # keep encoder_type var for logging on error later
    if t == "dino": return torch.hub.load('facebookresearch/dino:main','dino_resnet50')
    if t.startswith("dinov3"):
        enc_dir = "../encoders/"
        weights = resolve_unique_model_file(enc_dir, encoder_type)
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
       or t in {"unimodel", "uni2", "uni2-h", "uni2h", "uni"}:
        base_pre, norm = _imagenet_parts(resize_size)
    else:
        raise RuntimeError(f"Cannot deduct mean/std for {encoder_type}")

    return _build_pipeline(base_pre, norm, extra=extra, place=place)