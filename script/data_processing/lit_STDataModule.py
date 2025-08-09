from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import lightning as L
from script.data_processing.image_transforms import get_transforms
from script.data_processing.data_loader import get_dataset


def get_data_module(cfg):
    return STDataModule(cfg)


class STDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        # Save the entire config for easy access
        self.cfg = cfg

        # Set up transforms based on config
        self.transforms = get_transforms(cfg)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        # Determine max_len for debug mode
        max_len = 100 if self.cfg.get('debug', False) else None

        if stage in (None, 'fit'):
            self.train_dataset = get_dataset(
                self.cfg['data_dir'],
                genes=self.cfg['genes'],
                samples=self.cfg['train_samples'],
                transforms=self.transforms,
                bins=self.cfg.get("bins", 0),
                gene_data_filename=self.cfg['gene_data_filename'],
                max_len=max_len,
                lds_smoothing_csv=self.cfg.get("lds_weight_csv", None)
            )
            self.val_dataset = get_dataset(
                self.cfg['data_dir'],
                genes=self.cfg['genes'],
                samples=self.cfg['val_samples'],
                transforms=self.transforms,
                bins=self.cfg.get("bins", 0),
                gene_data_filename=self.cfg['gene_data_filename'],
                max_len=max_len,
                lds_smoothing_csv=self.cfg.get("lds_weight_csv", None)
            )
        if stage in (None, 'test') and self.cfg.get('test_samples'):
            self.test_dataset = get_dataset(
                self.cfg['data_dir'],
                genes=self.cfg['genes'],
                samples=self.cfg['test_samples'],
                transforms=self.transforms,
                bins=self.cfg.get("bins", 0),
                gene_data_filename=self.cfg['gene_data_filename'],
                max_len=max_len,
                lds_smoothing_csv=self.cfg.get("lds_weight_csv", None)
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=True,
            num_workers=self.cfg.get('num_workers', 0),
            pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=False,
            num_workers=self.cfg.get('num_workers', 0),
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=False,
            num_workers=self.cfg.get('num_workers', 0),
            pin_memory=False
        )

    def free_memory(self):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
