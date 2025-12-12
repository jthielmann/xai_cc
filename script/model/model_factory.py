import os

import torch
import torch.nn as nn
import torchvision.models as models
try:
    from torchvision.transforms import v2  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback path for older torchvision
    from torchvision import transforms as _transforms
    from torchvision.transforms import functional as _F

    class _ToImage:
        def __call__(self, img):
            if isinstance(img, torch.Tensor):
                return img
            return _F.to_tensor(img)

    class _ToDtype:
        def __init__(self, dtype, scale: bool = False):
            self.dtype = dtype
            self.scale = scale

        def __call__(self, img):
            tensor = img.to(self.dtype)
            if self.scale and torch.is_floating_point(tensor) and tensor.max() > 1:
                tensor = tensor / 255
            return tensor

    class _Resize:
        def __init__(self, size, antialias: bool = True):
            self._resize = _transforms.Resize(size)

        def __call__(self, img):
            return self._resize(img)

    class _V2Compat:
        Compose = _transforms.Compose
        Normalize = _transforms.Normalize
        Resize = _Resize
        ToImage = _ToImage
        ToDtype = _ToDtype

    v2 = _V2Compat()
from script.configs.normalization import IMAGENET_MEAN, IMAGENET_STD

from script.configs.config_factory import get_dataset_cfg
import timm
from script.model.dinov3_local import load_dinov3_local
from typing import Tuple

from typing import Iterable, Sequence, Callable

# resolver.py (or inline in your loader code)
from pathlib import Path
from typing import Iterable


def resolve_unique_model_file(
        encoder_type: str,
        encoders_dir = "../encoders",
) -> Path:
    # Map: hub callable name -> checkpoint filename
    DINOv3_FILEMAP = {
        "dinov3_convnext_base": "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
        "dinov3_convnext_large": "dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth",
        "dinov3_convnext_tiny": "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
        "dinov3_vit7b16": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
        "dinov3_vith16plus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
        "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    }
    alias = {"dinov3": "dinov3_vitb16"}

    t = (encoder_type or "").strip().lower()
    t = alias.get(t, t)

    if t in DINOv3_FILEMAP:
        base = Path(encoders_dir)
        path = base / DINOv3_FILEMAP[t]
        if not path.exists():
            raise FileNotFoundError(f"Resolved '{t}' to '{path.name}', but it does not exist in {base}.")
        return path

    for key, value in DINOv3_FILEMAP.items():
        if key in encoder_type:
            return Path(encoders_dir, value)
    raise RuntimeError(f"{encoder_type} not in {DINOv3_FILEMAP}")


def get_encoder(encoder_type: str) -> nn.Module:
    t = encoder_type.lower()  # keep encoder_type var for logging on error later
    if t == "dino": return torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    if t.startswith("dinov3"):
        return load_dinov3_local(t, "../encoders")
    if t == "resnet50random": return models.resnet50(weights=False)
    if t == "resnet50imagenet": return models.resnet50(weights="IMAGENET1K_V2")
    # Fix logic: ensure we only match UNI variants explicitly
    if "uni" in t: return load_uni_model()
    raise ValueError(f"Unknown encoder {encoder_type}")


def _extract_features(encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(encoder, "forward_features"):
        return encoder.forward_features(x)
    return encoder(x)


def infer_encoder_out_dim(
    encoder: nn.Module,
    input_size: Tuple[int, int, int] = (3, 224, 224),
    device = None,
    vit_pooling: str = "mean_patch",
) -> int:
    # Fast-path: many backbones expose feature dims directly
    for attr in ("embed_dim", "num_features"):
        val = getattr(encoder, attr, None)
        if isinstance(val, (int, float)):
            return int(val)


    # keep pre-trainer checks on CPU by restoring encoder device after probing
    # why: avoid accidental GPU placement before Lightning handles devices

    # Select accelerator device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Capture original device to restore after probing
    p0 = next(encoder.parameters(), None)
    if p0 is None:
        raise RuntimeError("Encoder has no parameters; cannot determine original device")
    orig_device = p0.device

    # Move encoder to device if needed and run minimal dummy forward
    was_training = encoder.training
    encoder = encoder.to(device)
    encoder.eval()
    dummy = torch.zeros(1, *input_size, device=device)
    with torch.no_grad():
        out = _extract_features(encoder, dummy)

    # Unwrap common containers
    if isinstance(out, (list, tuple)) and len(out) > 0:
        out = out[0]
    if hasattr(out, "last_hidden_state"):
        out = out.last_hidden_state

    out = normalize_encoder_out(out, vit_pooling=vit_pooling)
    if out.ndim != 2:
        raise RuntimeError(f"normalized encoder output must be 2D, got {tuple(out.shape)}")
    feat_dim = int(out.size(1))

    # Restore original device and training mode
    encoder.to(orig_device)
    if was_training:
        encoder.train()
    return int(feat_dim)


def normalize_encoder_out(z: torch.Tensor, *, vit_pooling: str = "mean_patch") -> torch.Tensor:
    if z.ndim == 4:
        return z.mean(dim=(2, 3))
    if z.ndim == 3:
        mode = str(vit_pooling)
        if mode == "mean_patch":
            if z.size(1) <= 1:
                raise ValueError(f"mean_patch requires CLS+patches; got tokens={z.size(1)}")
            return z[:, 1:, :].mean(dim=1)
        if mode == "mean_all":
            return z.mean(dim=1)
        if mode == "cls":
            return z[:, 0, :]
        raise ValueError(f"unknown vit_pooling={vit_pooling!r}")
    if z.ndim == 2:
        return z
    raise RuntimeError(f"Unsupported encoder output shape {tuple(z.shape)}")


def build_model(**kwargs):
    return timm.create_model(**kwargs)


# similar to https://huggingface.co/MahmoodLab/UNI2-h
def load_uni_model():
    model_file = "../encoders/UNI2-h_state.pt"  # local cache of the weights
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
    state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
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
    norm = v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return pre, norm


def _build_pipeline(base_pre: Sequence[Callable], norm: Callable,
                    extra: Iterable[Callable] = None,
                    place: str = "pre_norm") -> v2.Compose:
    extra_list = [] if extra is None else list(extra)
    if place == "pre_norm":
        steps = [*base_pre, *extra_list, norm]
    elif place == "post_norm":
        steps = [*base_pre, norm, *extra_list]
    else:
        raise ValueError(f"Unknown place='{place}' (use 'pre_norm' or 'post_norm').")
    return v2.Compose(steps)


def get_encoder_transforms(encoder_type: str,
                           resize_size: int = 224,
                           extra: Iterable[Callable] = None,
                           place: str = "pre_norm") -> v2.Compose:
    """
    `extra`: optional iterable of v2 transforms to insert (default before Normalize).
    Typical occlusion/erasing goes BEFORE Normalize and AFTER ToDtype.
    """
    if not isinstance(encoder_type, str) or not encoder_type.strip():
        raise ValueError(f"invalid encoder_type: {encoder_type!r}")
    t = encoder_type.lower()

    # All listed encoders here use ImageNet mean/std
    if t == "dino" or t.startswith("resnet") or t.startswith("dinov3") \
            or t.startswith("uni"):
        base_pre, norm = _imagenet_parts(resize_size)
    else:
        raise RuntimeError(f"Cannot deduct mean/std for {encoder_type}")

    return _build_pipeline(base_pre, norm, extra=extra, place=place)
