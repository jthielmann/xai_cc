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

        # HF processor (resize/normalize) – keep toggle so you can use your own transforms instead
        self.processor = AutoImageProcessor.from_pretrained(model_id)

        # Student / teacher ViT backbones
        self.student_backbone = Dinov3Model.from_pretrained(model_id)
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        deactivate_requires_grad(self.teacher_backbone)

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

    def on_fit_start(self):
        os.makedirs(self.config.get('out_path', '.'), exist_ok=True)

    def update_lr(self, lr):
        self.lr = lr

    def forward(self, pixel_values):
        y = self._encode_cls(pixel_values, self.student_backbone)
        z = self.student_head(y)
        return z

    def forward_teacher(self, pixel_values):
        y = self._encode_cls(pixel_values, self.teacher_backbone)
        z = self.teacher_head(y)
        return z

    def common_step(self, batch, batch_idx):
        # momentum schedule (unchanged)
        momentum = cosine_schedule(
            step=self.current_epoch,
            max_steps=self.config['epochs'],
            end_value=0.996,
            start_value=1.0
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
                # If v is a PIL image or tensor (C,H,W) float in [0,1], processor can handle it.
                pv = self.processor(images=v, return_tensors="pt")["pixel_values"].to(self.device)
            else:
                # If you already normalize in your dataloader/transforms, just ensure it's a float tensor (B,C,H,W) on device
                pv = v if v.dim() == 4 else v.unsqueeze(0)
                pv = pv.to(self.device)
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
        wd = self.config.get('weight_decay', 0.0)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=wd)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
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
            wandb.run.summary['best_val_loss'] = self.best_loss
            wandb.run.summary['best_val_epoch'] = self.current_epoch
            best = os.path.join(self.config.get('out_path', '.'), 'best_model.pth')
            torch.save(self.state_dict(), best)
            wandb.log({'epoch': self.current_epoch})

    def set_num_training_batches(self, num_batches):
        self.num_training_batches = num_batches
