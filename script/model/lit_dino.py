import os
import copy

import torch
import torch.nn as nn
import torchvision
import lightning as L
from torch.optim.lr_scheduler import OneCycleLR

import wandb
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from transformers import AutoImageProcessor, Dinov3Model

class DINO(L.LightningModule):
    def __init__(self, dino_config, lr=0.001):
        super().__init__()
        self.config = dino_config
        self.lr = lr

        # --- NEW: DINOv3 backbone setup ---
        # config knobs you can pass in:
        # self.config.get("model_id", "facebook/dinov3-vitb16-pretrain-lvd1689m")
        # self.config.get("use_hf_normalize", True)

        model_id = self.config.get(
            "model_id", "facebook/dinov3-vitb16-pretrain-lvd1689m"
        )
        self.use_hf_normalize = bool(self.config.get("use_hf_normalize", True))
        # Backbone training knobs
        self.grad_checkpointing = bool(self.config.get("grad_checkpointing", True))
        # Unfreeze last N transformer blocks (0 = none; -1 = all)
        self.unfreeze_last_n = int(self.config.get("unfreeze_last_n", -1))
        # Backbone LR multiplier (default 1.0 to avoid behavior change)
        self.backbone_lr_mult = float(self.config.get("backbone_lr_mult", 1.0))
        # Optional explicit backbone LR overrides multiplier
        self.backbone_lr = self.config.get("backbone_lr", None)

        # HF processor (resize/normalize) – keep toggle so you can use your own transforms instead
        self.processor = AutoImageProcessor.from_pretrained(model_id)

        # Student / teacher ViT backbones
        self.student_backbone = Dinov3Model.from_pretrained(model_id)
        # Enable gradient checkpointing to save VRAM
        if self.grad_checkpointing and hasattr(self.student_backbone, "gradient_checkpointing_enable"):
            try:
                self.student_backbone.gradient_checkpointing_enable()
            except Exception:
                pass
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        deactivate_requires_grad(self.teacher_backbone)

        # Optionally restrict trainable layers of the student backbone
        self._configure_backbone_trainability(self.unfreeze_last_n)

        # Hidden size drives head input dim (e.g., 768 for ViT-B/16)
        hidden = self.student_backbone.config.hidden_size

        # Projection heads (same output_dim as before)
        self.student_head  = DINOProjectionHead(hidden, 512, 64, 2048, freeze_last_layer=1)
        self.teacher_head  = DINOProjectionHead(hidden, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_head)

        # Loss unchanged
        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)
        self.best_loss = torch.tensor(float('inf'))
        self.current_loss = torch.tensor(0.)
        self.num_training_batches = 0

    def _encode_cls(self, pixel_values, model: Dinov3Model):
        out = model(pixel_values=pixel_values)            # returns BaseModelOutputWithPooling
        tokens = out.last_hidden_state                    # (B, 1+N, D)
        cls = tokens[:, 0]                                # (B, D)
        return cls

    def _find_transformer_layers(self, model: nn.Module):
        # Try common HF ViT/DINOv3 layer containers
        candidates = [
            "vit.encoder.layer",
            "dinov3.encoder.layer",
            "encoder.layer",
            "blocks",
            "layers",
        ]
        for path in candidates:
            obj = model
            ok = True
            for attr in path.split('.'):
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok and isinstance(obj, (nn.ModuleList, list, tuple)) and len(obj) > 0:
                return obj
        # Fallback: scan named_modules for a ModuleList that looks like transformer blocks
        for name, mod in model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 0 and any(k in name.lower() for k in ["layer", "block", "encoder"]):
                return mod
        return None

    def _configure_backbone_trainability(self, unfreeze_last_n: int):
        # Default: train all; allow selectively unfreezing last N blocks
        if unfreeze_last_n is None:
            return
        if unfreeze_last_n == -1:
            # train all backbone params
            for p in self.student_backbone.parameters():
                p.requires_grad = True
            return
        # freeze all first
        for p in self.student_backbone.parameters():
            p.requires_grad = False
        # unfreeze last N blocks if we can find them
        blocks = self._find_transformer_layers(self.student_backbone)
        if isinstance(blocks, (nn.ModuleList, list, tuple)) and len(blocks) > 0 and unfreeze_last_n > 0:
            n = min(unfreeze_last_n, len(blocks))
            for blk in blocks[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True
        # Also allow final layernorm/head norms to train for stability
        for name, module in self.student_backbone.named_modules():
            if any(k in name.lower() for k in ["layernorm", "ln", "final_norm"]):
                for p in module.parameters():
                    p.requires_grad = True

    def on_fit_start(self):
        os.makedirs(self.config.get('out_path', '.'), exist_ok=True)

        # ---- total_steps for per-step cosine schedules ----
        # Prefer Lightning's estimate; fallback to epochs * num_training_batches
        if hasattr(self.trainer, "estimated_stepping_batches") and self.trainer.estimated_stepping_batches:
            self.total_steps = int(self.trainer.estimated_stepping_batches)
        else:
            self.total_steps = int(self.config['epochs'] * max(1, self.num_training_batches))

    def update_lr(self, lr):
        self.lr = lr

    def forward(self, pixel_values):
        y = self._encode_cls(pixel_values, self.student_backbone)
        z = self.student_head(y)
        return z

    def forward_teacher(self, pixel_values):
        with torch.no_grad():
            y = self._encode_cls(pixel_values, self.teacher_backbone)
        return self.teacher_head(y)

    def common_step(self, batch, batch_idx):
        # Step-based EMA momentum
        step_now = min(self.global_step, max(0, self.total_steps - 1))
        momentum = cosine_schedule(
            step=step_now,
            max_steps=self.total_steps,
            start_value=1.0,
            end_value=0.996,
        )
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        # Views handling: support (views, ...) or plain views
        views_in = batch[0] if (isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], (list, tuple))) else batch
        views_in = views_in if isinstance(views_in, (list, tuple)) else [views_in]

        # If you rely on HF processor normalization:
        # views_in are assumed to be lists of PIL images or tensors in [0,1] (C,H,W)
        views = []
        for v in views_in:
            if self.use_hf_normalize:
                if torch.is_tensor(v) and v.dim() == 4:           # (B,C,H,W) already
                    pv = v.to(self.device)                         # assume you normalized earlier
                else:
                    pv = self.processor(images=v, return_tensors="pt")["pixel_values"].to(self.device)
            else:
                pv = (v if (torch.is_tensor(v) and v.dim()==4) else v.unsqueeze(0)).to(self.device)
            views.append(pv)

        # teacher: first 2 global crops; student: all crops
        teacher_out = [self.forward_teacher(v) for v in views[:2]]
        student_out = [self.forward(v) for v in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True)
        self.current_loss += loss
        return loss

    def on_after_backward(self):
        # freeze last layer gradients periodically
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        if not hasattr(self, 'num_training_batches') or self.num_training_batches <= 0:
            raise ValueError(
                '`num_training_batches` must be set before configuring optimizers'
            )
        wd = float(self.config.get('weight_decay', 0.0))

        # Build optimizer parameter groups: bias/norm no-decay, lower LR for backbone
        def split_decay_groups(module: nn.Module):
            decay, no_decay = [], []
            seen = set()
            norm_types = (
                nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d,
                nn.InstanceNorm2d, nn.InstanceNorm3d,
            )
            for m in module.modules():
                for name, p in getattr(m, 'named_parameters', lambda recurse=False: [])(recurse=False):
                    # full param object identity check to avoid duplicates
                    if not p.requires_grad or id(p) in seen:
                        continue
                    seen.add(id(p))
                    if name.endswith('bias') or isinstance(m, norm_types):
                        no_decay.append(p)
                    else:
                        decay.append(p)
            return decay, no_decay

        # Backbone groups (trainable only)
        bb_decay, bb_no_decay = split_decay_groups(self.student_backbone)
        # Head groups
        head_decay, head_no_decay = split_decay_groups(self.student_head)

        # Compute backbone LR
        backbone_lr = float(self.backbone_lr) if self.backbone_lr is not None else float(self.lr) * float(self.backbone_lr_mult)

        param_groups = []
        if bb_decay:
            param_groups.append({"params": bb_decay, "lr": backbone_lr, "weight_decay": wd})
        if bb_no_decay:
            param_groups.append({"params": bb_no_decay, "lr": backbone_lr, "weight_decay": 0.0})
        if head_decay:
            param_groups.append({"params": head_decay, "lr": float(self.lr), "weight_decay": wd})
        if head_no_decay:
            param_groups.append({"params": head_no_decay, "lr": float(self.lr), "weight_decay": 0.0})

        # Fallback in unlikely case groups are empty
        if not param_groups:
            param_groups = [{"params": [p for p in self.parameters() if p.requires_grad], "lr": float(self.lr), "weight_decay": wd}]

        optimizer = torch.optim.AdamW(param_groups)
        # Respect per-group LR by passing a list to max_lr
        max_lrs = [g.get('lr', float(self.lr)) for g in param_groups]
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            epochs=self.config['epochs'],
            steps_per_epoch=self.num_training_batches
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

    def on_train_epoch_end(self):
        if not self.config.get('debug'):
            latest_path = os.path.join(self.config.get('out_path', '.'), 'latest.pth')
            torch.save(self.state_dict(), latest_path)

    def on_validation_start(self):
        self.current_loss = torch.tensor(0.).to(self.device)

    def on_validation_end(self):
        # save latest checkpoint
        if not self.config.get('debug'):
            latest = os.path.join(self.config.get('out_path', '.'), 'latest.pth')
            torch.save(self.state_dict(), latest)

        # save best model if improved
        if self.current_loss < self.best_loss:
            self.best_loss = self.current_loss
            if wandb.run:
                wandb.run.summary['best_val_loss'] = float(self.best_loss)
                wandb.run.summary['best_val_epoch'] = int(self.current_epoch)
                wandb.log({'epoch': int(self.current_epoch)})
            best = os.path.join(self.config.get('out_path', '.'), 'best_model.pth')
            torch.save(self.state_dict(), best)

    def set_num_training_batches(self, num_batches):
        self.num_training_batches = num_batches
