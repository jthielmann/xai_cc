import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import lightning as L
from torch.utils.data import DataLoader

from script.model.lit_dino import DINO


def make_fake_multicrop_sample(gsize=96, lsize=64, n_local=2):
    g1 = torch.rand(3, gsize, gsize)
    g2 = torch.rand(3, gsize, gsize)
    locals_ = [torch.rand(3, lsize, lsize) for _ in range(n_local)]
    return [g1, g2] + locals_


class RandomMultiCropDataset(torch.utils.data.Dataset):
    def __init__(self, length=32):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return list of unbatched views; DataLoader will stack to (B,C,H,W)
        return make_fake_multicrop_sample()


def main():
    cfg = {
        'epochs': 1,
        'use_dummy_backbone': True,
        'dummy_hidden_size': 128,
        'head_hidden_dim': 128,
        'head_bottleneck_dim': 64,
        'head_output_dim': 256,
        'freeze_last_layer_epochs': 1,
        'teacher_temp_warmup': 0.04,
        'teacher_temp_final': 0.07,
        'warmup_teacher_temp_epochs': 1,
        'student_temp': 0.1,
        'center_momentum': 0.9,
        'ema_momentum_start': 0.996,
        'ema_momentum_end': 0.9995,
        'out_path': '.',
        'debug': True,
    }

    model = DINO(cfg, lr=1e-3)

    ds_train = RandomMultiCropDataset(32)
    ds_val = RandomMultiCropDataset(16)
    dl_train = DataLoader(ds_train, batch_size=8, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=8)

    model.set_num_training_batches(len(dl_train))

    trainer = L.Trainer(max_epochs=1, accelerator='cpu', devices=1, log_every_n_steps=1, enable_checkpointing=False)
    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val)


if __name__ == '__main__':
    main()
