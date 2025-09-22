import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve()).split('/script/tests/')[0])

import torch
from script.model.lit_dino import DINO


def run():
    print("[optim] init DINO with LLRD + cosine weight decay")
    cfg = {
        'epochs': 1,
        'use_dummy_backbone': True,
        'dummy_hidden_size': 128,
        'head_output_dim': 256,
        'layer_lr_decay': 0.0,  # dummy backbone has no blocks; test backbone vs head lr split
        'backbone_lr_mult': 0.1,
        'weight_decay': 0.1,
        'weight_decay_cosine': True,
        'weight_decay_end': 0.0,
        'debug': True,
    }
    model = DINO(cfg)
    model.set_num_training_batches(4)
    model.update_lr(1e-3)
    opt_cfg = model.configure_optimizers()
    optim = opt_cfg['optimizer']
    lrs = sorted({round(g.get('lr', 0.0), 8) for g in optim.param_groups if len(g.get('params', [])) > 0}, reverse=True)
    print(f"[optim] param group LRs (desc): {lrs}")
    assert len(optim.param_groups) >= 3, "expected multiple param groups (backbone decay/no-decay + head)"

    # test weight decay schedule updates
    print("[optim] weight decay cosine schedule progression")
    # attach dummy trainer/global_step for schedule hook
    class _DummyTrainer:
        def __init__(self):
            self.global_step = 0
            self.optimizers = [optim]
            self.estimated_stepping_batches = None
    model._trainer = _DummyTrainer()
    model.on_before_optimizer_step(optim, optimizer_idx=0)
    wd0 = max(pg.get('weight_decay', 0.0) for pg in optim.param_groups)
    model._trainer.global_step = model.num_training_batches * cfg['epochs']
    model.on_before_optimizer_step(optim, optimizer_idx=0)
    wd1 = max(pg.get('weight_decay', 0.0) for pg in optim.param_groups)
    print(f"[optim] weight_decay: {wd0:.5f} -> {wd1:.5f}")
    assert wd1 <= wd0 + 1e-8, "expected non-increasing weight decay over cosine schedule"

    # fused adamw safe toggle
    print(f"[optim] use_fused_adamw on CUDA only (current cuda={torch.cuda.is_available()})")
    cfg['use_fused_adamw'] = True
    model2 = DINO(cfg)
    model2.set_num_training_batches(4)
    _ = model2.configure_optimizers()
    print("[optim] fused AdamW creation did not error (fallback safe)")


if __name__ == '__main__':
    run()
