import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from script.model.lit_dino import DINO


def run():
    print("[heads] init DINO with dummy backbone + custom head dims")
    cfg = {
        'epochs': 1,
        'use_dummy_backbone': True,
        'dummy_hidden_size': 128,
        'head_hidden_dim': 96,
        'head_bottleneck_dim': 64,
        'head_output_dim': 256,
        'freeze_last_layer_epochs': 1,
        'debug': True,
    }
    model = DINO(cfg)

    hidden = int(model.student_backbone.config.hidden_size)
    assert hidden == 128, f"hidden_size mismatch: {hidden} != 128"
    print(f"[heads] backbone hidden_size={hidden}")

    # verify loss output dim alignment
    loss_out_dim = int(model.criterion.center.shape[-1])
    assert loss_out_dim == cfg['head_output_dim'], f"DINOLoss output_dim {loss_out_dim} != head_output_dim {cfg['head_output_dim']}"
    print(f"[heads] DINOLoss output_dim={loss_out_dim} matches head_output_dim")

    # forward shapes
    x = torch.randn(4, 3, 96, 96)
    z = model(x)
    assert z.shape == (4, cfg['head_output_dim']), f"forward shape {tuple(z.shape)}"
    print(f"[heads] forward shape OK: {tuple(z.shape)}")

    # teacher eval mode
    assert not model.teacher_backbone.training and not model.teacher_head.training, "teacher modules should be eval()"
    print("[heads] teacher eval mode confirmed")


if __name__ == '__main__':
    run()

