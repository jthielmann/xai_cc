import os

import torch
import torch.nn as nn
import lightning as L
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
from torch.optim.lr_scheduler import OneCycleLR

from script.model.model_factory import get_encoder, infer_encoder_out_dim


class Autoencoder(L.LightningModule):
    """
    Simple autoencoder for regression tasks, using a encoder CNN encoder.

    Expected `config` keys:
      - encoder_type (str)
      - encoder_out_dim (int)
      - middle_layer_features (int)
      - ae_out_features (int)
      - learning_rate (float)
      - epochs (int)
      - freeze_encoder (bool)
      - out_path (str)
    """
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)

        self.encoder = get_encoder(config['encoder_type'])
        if config.get('freeze_encoder', False):
            for p in self.encoder.parameters():
                p.requires_grad = False

        in_dim = config['encoder_out_dim']
        hid_dim = config.get('middle_layer_features', 256)
        out_dim = config['ae_out_features']

        act = nn.LeakyReLU if config.get('use_leaky_relu', False) else nn.ReLU
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            act(),
            nn.Linear(hid_dim, out_dim)
        )

        self.loss_fn = nn.MSELoss()
        self.pearson = torchmetrics.PearsonCorrCoef(num_outputs=out_dim)
        self.best_loss = torch.tensor(float('inf'))
        self.num_training_batches = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_fit_start(self):
        os.makedirs(self.hparams.out_path, exist_ok=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)
        if not hasattr(self, 'val_yhats'):
            self.val_yhats, self.val_ys, self.val_loss = [], [], torch.tensor(0., device=self.device)
        self.val_yhats.append(y_hat)
        self.val_ys.append(y)
        self.val_loss += loss
        return loss

    def on_validation_epoch_end(self):
        # skip Lightning sanity check
        if not hasattr(self, '_sanity_done'):
            self._sanity_done = True
            return

        # aggregate
        yh = torch.cat(self.val_yhats)
        yt = torch.cat(self.val_ys)
        r = self.pearson(yh, yt)
        self.log('pearson', r)

        # checkpoint
        latest = os.path.join(self.hparams.out_path, 'latest.pth')
        torch.save(self.state_dict(), latest)
        if self.val_loss < self.best_loss:
            self.best_loss = self.val_loss
            best = os.path.join(self.hparams.out_path, 'best_model.pth')
            torch.save(self.state_dict(), best)

    def configure_optimizers(self):
        if self.num_training_batches <= 0:
            raise RuntimeError('`set_num_training_batches` must be called before configure_optimizers')
        params = [
            {'params': self.encoder.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.decoder.parameters(), 'lr': self.hparams.learning_rate},
        ]
        opt = optim.AdamW(params)
        sch = OneCycleLR(
            opt,
            max_lr=self.hparams.learning_rate,
            epochs=self.hparams.epochs,
            steps_per_epoch=self.num_training_batches
        )
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sch, 'interval': 'step'}}

    def set_num_training_batches(self, n: int):
        self.num_training_batches = n

class SparseAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_in = config['d_in']
        d_hidden = config['d_hidden']
        self.encoder = nn.Linear(d_in, d_hidden, bias=config.get('encoder_bias', False))
        self.topk_activation = TopKActivation(config['k'])
        # True weight tying: decode with encoder.weight.T and a separate bias
        self.decoder_bias = nn.Parameter(torch.zeros(d_in))
        self.last_sparse = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        sparse = self.topk_activation(encoded)
        self.last_sparse = sparse.detach()
        # Decode via tied weights (encoder.weight.T) and learned bias
        decoded = F.linear(sparse, self.encoder.weight.t(), self.decoder_bias)
        return decoded


class LitSparseAutoencoder(L.LightningModule):
    """
    LightningModule for training a SparseAutoencoder.
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.sae = SparseAutoencoder(config)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sae(x)

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        recon = self(x)
        loss = self.loss_fn(recon, x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        recon = self(x)
        loss = self.loss_fn(recon, x)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        encoded = self.sae.encoder(x)
        sparse = self.sae.topk_activation(encoded)
        return sparse

    def configure_optimizers(self):
        lr = float(self.hparams.get("learning_rate") or self.hparams.get("lr") or 1e-3)
        wd = float(self.hparams.get("weight_decay") or 1e-2)
        opt = optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        # OneCycleLR over full training
        sched = OneCycleLR(
            opt,
            max_lr=lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}




class LitSparseAutoencoder(L.LightningModule):
    """
    LightningModule for training a SparseAutoencoder.
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.sae = SparseAutoencoder(config)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sae(x)

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        recon = self(x)
        loss = self.loss_fn(recon, x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        recon = self(x)
        loss = self.loss_fn(recon, x)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr = float(self.hparams.get("learning_rate") or self.hparams.get("lr") or 1e-3)
        wd = float(self.hparams.get("weight_decay") or 1e-2)
        opt = optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        # OneCycleLR over full training
        sched = OneCycleLR(
            opt,
            max_lr=lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}



def get_autoencoder(config:dict):
    ae_type = config.get("ae_type")
    if ae_type == "sparse":
        return SparseAutoencoder(config)
    # elif ae_type == "denoising":
    #     return DenoisingAutoencoder(input_dim, hidden_dim)
    elif ae_type == "vanilla":
        return Autoencoder(config)
    else:
        raise ValueError(f"Unknown ae_type={ae_type}")
