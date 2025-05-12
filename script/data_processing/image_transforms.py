import importlib
v2_ready = importlib.util.find_spec("torchvision.transforms.v2")
if v2_ready is not None:
    import torchvision.transforms.v2 as transforms
else:
    import torchvision.transforms as transforms

import torch

def get_transforms(cfg, normalize=True):
    image_size = cfg["image_size"]

    mean = cfg["mean"]
    std = cfg["std"]
    if v2_ready is not None:
        ts = [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomRotation(90),
                    #transforms.GaussianNoise(), max sagt ohne
                    #transforms.ToTensor(),
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True)]
        if normalize:
            ts.append(transforms.Normalize(mean, std))
        image_transforms = transforms.Compose(ts)
    else:
        ts = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(90),
            # transforms.GaussianNoise(), max sagt ohne
            transforms.ToTensor(),
            # mean and std of the whole dataset
            transforms.Normalize(mean, std)]
        if normalize:
            ts.append(transforms.Normalize(mean, std))
        image_transforms = transforms.Compose(ts)
    return image_transforms