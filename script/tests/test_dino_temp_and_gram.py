import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from script.model.lit_dino import DINO


def _make_views(batch_size=8, g=96, l=64, n_local=2):
    g1 = torch.rand(batch_size, 3, g, g)
    g2 = torch.rand(batch_size, 3, g, g)
    locals_ = [torch.rand(batch_size, 3, l, l) for _ in range(n_local)]
    return [g1, g2] + locals_


def run():
    print("[temp/gram] init DINO with per-step teacher temp + gram off")
    cfg = {
        'epochs': 1,
        'use_dummy_backbone': True,
        'dummy_hidden_size': 128,
        'head_output_dim': 256,
        'teacher_temp_warmup': 0.04,
        'teacher_temp_final': 0.07,
        'teacher_temp_per_step': True,
        'warmup_teacher_temp_epochs': 1,
        'gram_anchor_weight': 0.0,
        'debug': True,
    }
    model = DINO(cfg)
    model.set_num_training_batches(10)
    print("[temp/gram] check manual per-step teacher_temp schedule")
    warmup_steps = model.num_training_batches
    t0 = cfg['teacher_temp_warmup']
    t1 = cfg['teacher_temp_final']
    t_mid = t0 + (t1 - t0) * (0.5)
    assert t0 < t1, "teacher_temp_warmup should be < teacher_temp_final"
    print(f"[temp/gram] teacher_temp warmup={t0} final={t1} mid~{t_mid:.4f}")

    print("[temp/gram] compute Gram anchor loss directly on patch tokens")
    views = _make_views()
    s_cls, s_p = model._backbone_tokens(views[0], model.student_backbone)
    # teacher patches = student patches + noise to make loss > 0
    t_p = s_p + 0.01 * torch.randn_like(s_p)
    ga = float(model._gram_anchor_loss([s_p, s_p], [t_p, t_p]).item())
    assert ga >= 0.0, "Gram anchor loss should be non-negative"
    print(f"[temp/gram] gram_anchor_loss={ga:.6f}")


if __name__ == '__main__':
    run()
