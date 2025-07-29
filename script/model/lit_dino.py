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


class DINO(L.LightningModule):
    """
    Self-supervised DINO model based on ResNet backbones.
    """
    def __init__(self, dino_config, lr=0.001):
        super().__init__()
        self.config = dino_config
        self.lr = lr

        backbone_name = self.config['encoder_type']
        if backbone_name == "resnet18":
            resnet = torchvision.models.resnet18(encoder=False)
            input_dim = 512
        elif backbone_name == "resnet50":
            resnet = torchvision.models.resnet50(encoder=False)
            input_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone {backbone_name}")

        # student and teacher networks
        self.student_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.student_head = DINOProjectionHead(input_dim, 512, 64, 2048, freeze_last_layer=1)
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)

        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)
        self.best_loss = torch.tensor(float('inf'))
        self.current_loss = torch.tensor(0.)
        self.num_training_batches = 0

    def on_fit_start(self):
        os.makedirs(self.config['out_path'], exist_ok=True)

    def update_lr(self, lr):
        self.lr = lr

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def common_step(self, batch, batch_idx):
        # momentum schedule
        momentum = cosine_schedule(
            step=self.current_epoch,
            max_steps=self.config['epochs'],
            end_value=0.996,
            start_value=1.0
        )
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        views = [v.to(self.device) for v in batch[0]]
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
            latest_path = os.path.join(self.config['out_path'], 'latest.pth')
            torch.save(self.state_dict(), latest_path)

    def on_validation_start(self):
        self.current_loss = torch.tensor(0.).to(self.device)

    def on_validation_end(self):
        # save latest checkpoint
        if not self.config.get('debug'):
            latest = os.path.join(self.config['out_path'], 'latest.pth')
            torch.save(self.state_dict(), latest)

        # save best model if improved
        if self.current_loss < self.best_loss:
            self.best_loss = self.current_loss
            wandb.run.summary['best_val_loss'] = self.best_loss
            wandb.run.summary['best_val_epoch'] = self.current_epoch
            best = os.path.join(self.config['out_path'], 'best_model.pth')
            torch.save(self.state_dict(), best)
            wandb.log({'epoch': self.current_epoch})

    def set_num_training_batches(self, num_batches):
        self.num_training_batches = num_batches
