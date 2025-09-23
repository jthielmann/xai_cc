import importlib
v2_ready = importlib.util.find_spec("torchvision.transforms.v2")
if v2_ready is not None:
    import torchvision.transforms.v2 as transforms
else:
    import torchvision.transforms as transforms

import torch
import torchvision
from torchvision.transforms import v2

from torchvision.transforms import v2 as T
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_train_transforms(image_size: int = 256, frozen_encoder: bool = False):
    # Geometric augs first
    augs = [
        T.RandomResizedCrop(image_size, scale=(0.75, 1.0), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        # 90° rotations (keeps tissue semantics):
        T.RandomChoice([T.RandomRotation([0,0]),
                        T.RandomRotation([90,90]),
                        T.RandomRotation([180,180]),
                        T.RandomRotation([270,270])], p=[0.25,0.25,0.25,0.25]),
        # Small affine jitter (tiny translate/scale):
        T.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.97, 1.03)),
    ]

    if not frozen_encoder:
        augs.append(T.RandomApply([T.ColorJitter(0.05, 0.05, 0.03, 0.01)], p=0.2))

    return T.Compose([
        T.ToImage(),
        *augs,
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_eval_transforms(image_size: int = 256):
    # Simple deterministic preprocessing for validation/test
    return T.Compose([
        T.ToImage(),
        T.Resize((image_size, image_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])





def get_transforms_dinov3(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])