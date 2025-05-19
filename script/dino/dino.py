# adapted from https://docs.lightly.ai/self-supervised-learning/examples/dino.html#dino
# This example requires the following dependencies to be installed:
# pip install lightly

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

from script.model.lit_model import DINO
from lightly.transforms.dino_transform import DINOTransform
from script.data_processing.data_loader import get_dino_dataset
import lightning as L
import torch
import torchvision
from script.data_processing.process_csv import get_dino_csv
from lightning.pytorch.loggers import WandbLogger
from script.configs.dino_config import dino_config


def train_model_sweep_dino(config=None):
    model = DINO(config)

    transform = DINOTransform()
    # we ignore object detection annotations by setting target_transform to return 0


    def target_transform(t):
        return 0

    file_path_train, file_path_val = get_dino_csv(0.8, "../data/NCT-CRC-HE-100K/")

    dataset_name = "dino_dataset"
    if dataset_name == "pascal_voc":
        train_dataset = torchvision.datasets.VOCDetection("datasets/pascal_voc", download=True, transform=transform, target_transform=target_transform)
        val_dataset = None
    elif dataset_name == "dino_dataset":
        from script.configs.dino_config import bins

        train_dataset = get_dino_dataset(csv_path=file_path_train, transforms=transform, max_len=100 if dino_config["debug"] else None, bins=bins, device_handling=False)
        val_dataset   = get_dino_dataset(csv_path=file_path_val,   transforms=transform, max_len=100 if dino_config["debug"] else None, bins=bins, device_handling=False)
    else:
        exit("Dataset not supported")

    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=0)
    dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=0)

    accelerator = "gpu" if torch.cuda.is_available() or torch.backends.mps.is_available() else exit("No GPU available")
    trainer = L.Trainer(max_epochs=10, devices=1, accelerator=accelerator, logger=WandbLogger(project=dino_config["project"]))
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)


if __name__ == '__main__':
    train_model_sweep_dino()