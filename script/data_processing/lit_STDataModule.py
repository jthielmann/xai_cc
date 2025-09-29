from torch.utils.data import DataLoader
import os
import torch
from torchvision import transforms
import lightning as L
from script.data_processing.transforms import build_transforms
from script.data_processing.data_loader import get_dataset, get_dataset_single_file


def get_data_module(cfg):
    return STDataModule(cfg)


class STDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        # Save the entire config for easy access
        self.cfg = cfg

        # Set up transforms based on config (train/eval)
        _t = build_transforms(cfg)
        self.transforms_train = _t["train"]
        self.transforms_eval = _t["eval"]
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        # Determine max_len for debug mode
        max_len = 100 if self.cfg.get('debug', False) else None

        # Determine if WMSE is selected; only then attach LDS weights
        loss_switch = str(self.cfg.get('loss_fn_switch', '')).lower()
        wmse_selected = loss_switch in {"wmse", "weighted mse"}
        if wmse_selected and not self.cfg.get('lds_weight_csv'):
            raise ValueError("WMSE selected but 'lds_weight_csv' is not set in config.")
        lds_csv = self.cfg.get('lds_weight_csv', None) if wmse_selected else None

        if stage in (None, 'fit'):
            # Validate mutually exclusive CSV config
            if self.cfg.get('single_csv_path') and (
                self.cfg.get('train_csv_path') or self.cfg.get('val_csv_path') or self.cfg.get('test_csv_path')
            ):
                raise ValueError("Provide either 'single_csv_path' or split-specific CSVs ('train_csv_path'/'val_csv_path'/'test_csv_path'), not both.")

            if self.cfg.get('single_csv_path'):
                scp = self.cfg['single_csv_path']
                # allow CSV path relative to data_dir
                if not os.path.isabs(scp) and self.cfg.get('data_dir'):
                    cand = os.path.join(self.cfg['data_dir'], scp)
                    csv_path = cand if os.path.isfile(cand) else scp
                else:
                    csv_path = scp
                self.train_dataset = get_dataset_single_file(
                    csv_path=csv_path,
                    data_dir=self.cfg.get('data_dir'),
                    genes=self.cfg['genes'],
                    transforms=self.transforms_train,
                    bins=self.cfg.get("bins", 0),
                    max_len=max_len,
                    lds_smoothing_csv=lds_csv,
                    tile_subdir=self.cfg.get('tile_subdir'),
                    split='train',
                    split_col_name=self.cfg.get('split_col_name', 'split')
                )
            elif self.cfg.get('train_csv_path'):
                self.train_dataset = get_dataset_single_file(
                    csv_path=self.cfg['train_csv_path'],
                    data_dir=self.cfg.get('data_dir'),
                    genes=self.cfg['genes'],
                    transforms=self.transforms_train,
                    bins=self.cfg.get("bins", 0),
                    max_len=max_len,
                    lds_smoothing_csv=lds_csv,
                    tile_subdir=self.cfg.get('tile_subdir')
                )
            else:
                self.train_dataset = get_dataset(
                    self.cfg['data_dir'],
                    genes=self.cfg['genes'],
                    samples=self.cfg['train_samples'],
                    transforms=self.transforms_train,
                    bins=self.cfg.get("bins", 0),
                    gene_data_filename=self.cfg['gene_data_filename'],
                    meta_data_dir=self.cfg.get('meta_data_dir', '/meta_data/'),
                    max_len=max_len,
                    lds_smoothing_csv=lds_csv
                )

            if self.cfg.get('single_csv_path'):
                scp = self.cfg['single_csv_path']
                if not os.path.isabs(scp) and self.cfg.get('data_dir'):
                    cand = os.path.join(self.cfg['data_dir'], scp)
                    csv_path = cand if os.path.isfile(cand) else scp
                else:
                    csv_path = scp
                self.val_dataset = get_dataset_single_file(
                    csv_path=csv_path,
                    data_dir=self.cfg.get('data_dir'),
                    genes=self.cfg['genes'],
                    transforms=self.transforms_eval,
                    bins=self.cfg.get("bins", 0),
                    max_len=max_len,
                    lds_smoothing_csv=lds_csv,
                    tile_subdir=self.cfg.get('tile_subdir'),
                    split='val',
                    split_col_name=self.cfg.get('split_col_name', 'split')
                )
            elif self.cfg.get('val_csv_path'):
                self.val_dataset = get_dataset_single_file(
                    csv_path=self.cfg['val_csv_path'],
                    data_dir=self.cfg.get('data_dir'),
                    genes=self.cfg['genes'],
                    transforms=self.transforms_eval,
                    bins=self.cfg.get("bins", 0),
                    max_len=max_len,
                    lds_smoothing_csv=lds_csv,
                    tile_subdir=self.cfg.get('tile_subdir')
                )
            else:
                self.val_dataset = get_dataset(
                    self.cfg['data_dir'],
                    genes=self.cfg['genes'],
                    samples=self.cfg['val_samples'],
                    transforms=self.transforms_eval,
                    bins=self.cfg.get("bins", 0),
                    gene_data_filename=self.cfg['gene_data_filename'],
                    meta_data_dir=self.cfg.get('meta_data_dir', '/meta_data/'),
                    max_len=max_len,
                    lds_smoothing_csv=lds_csv
                )
        if stage in (None, 'test'):
            if self.cfg.get('single_csv_path'):
                scp = self.cfg['single_csv_path']
                if not os.path.isabs(scp) and self.cfg.get('data_dir'):
                    cand = os.path.join(self.cfg['data_dir'], scp)
                    csv_path = cand if os.path.isfile(cand) else scp
                else:
                    csv_path = scp
                self.test_dataset = get_dataset_single_file(
                    csv_path=csv_path,
                    data_dir=self.cfg.get('data_dir'),
                    genes=self.cfg['genes'],
                    transforms=self.transforms_eval,
                    bins=self.cfg.get("bins", 0),
                    max_len=max_len,
                    lds_smoothing_csv=lds_csv,
                    tile_subdir=self.cfg.get('tile_subdir'),
                    split='test',
                    split_col_name=self.cfg.get('split_col_name', 'split')
                )
            elif self.cfg.get('test_csv_path'):
                self.test_dataset = get_dataset_single_file(
                    csv_path=self.cfg['test_csv_path'],
                    data_dir=self.cfg.get('data_dir'),
                    genes=self.cfg['genes'],
                    transforms=self.transforms_eval,
                    bins=self.cfg.get("bins", 0),
                    max_len=max_len,
                    lds_smoothing_csv=lds_csv,
                    tile_subdir=self.cfg.get('tile_subdir')
                )
            else:
                if self.cfg.get('test_samples'):
                    self.test_dataset = get_dataset(
                        self.cfg['data_dir'],
                        genes=self.cfg['genes'],
                        samples=self.cfg['test_samples'],
                        transforms=self.transforms_eval,
                        bins=self.cfg.get("bins", 0),
                        gene_data_filename=self.cfg['gene_data_filename'],
                        meta_data_dir=self.cfg.get('meta_data_dir', '/meta_data/'),
                        max_len=max_len,
                        lds_smoothing_csv=lds_csv
                    )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=True,
            num_workers=self.cfg.get('num_workers', 0),
            pin_memory=torch.cuda.is_available()
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=False,
            num_workers=self.cfg.get('num_workers', 0),
            pin_memory=torch.cuda.is_available()
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=False,
            num_workers=self.cfg.get('num_workers', 0),
            pin_memory=torch.cuda.is_available()
        ) if self.test_dataset else None

    def free_memory(self):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
