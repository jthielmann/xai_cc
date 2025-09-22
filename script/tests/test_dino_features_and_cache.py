import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import torch
from torch.utils.data import DataLoader, Dataset
from script.model.lit_dino import DINO


class RandomImageDataset(Dataset):
    def __init__(self, n=10, size=96):
        self.n = n
        self.size = size
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        return torch.rand(3, self.size, self.size)


def run():
    print("[features] encode and cache with dummy backbone")
    cfg = {
        'epochs': 1,
        'use_dummy_backbone': True,
        'dummy_hidden_size': 128,
        'debug': True,
    }
    model = DINO(cfg)
    ds = RandomImageDataset(n=8, size=96)
    dl = DataLoader(ds, batch_size=4)
    f_cls = model.encode_features(next(iter(dl)), pool='cls')
    f_mean = model.encode_features(next(iter(dl)), pool='mean')
    print(f"[features] cls={tuple(f_cls.shape)} mean={tuple(f_mean.shape)}")
    assert f_cls.shape[1] == model.student_backbone.config.hidden_size
    assert f_mean.shape[1] == model.student_backbone.config.hidden_size

    out_dir = Path(__file__).resolve().parent / "_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "features.pt"
    path = model.cache_features(dl, str(out_file), pool='cls', dtype='float16')
    assert os.path.exists(path), f"cache file not found: {path}"
    print(f"[features] cached -> {path}")


if __name__ == '__main__':
    run()

