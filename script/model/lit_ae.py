import os

import torch
import torch.nn as nn
import lightning as L
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
from torch.optim.lr_scheduler import OneCycleLR

from script.model.model_factory import get_encoder, infer_encoder_out_dim


class TopKActivation(nn.Module):
    """
    Activation layer that retains only the top-k values along the last dimension.

    If `allow_negative_topk` is False, the selection and outputs are based on
    ReLU(x), i.e., negative activations are zeroed and never selected.
    """
    def __init__(self, k: int, allow_negative_topk: bool = False):
        super().__init__()
        self.k = k
        self.allow_negative_topk = allow_negative_topk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = x if self.allow_negative_topk else F.relu(x)
        # Handle small dimensions gracefully
        k = min(self.k, scores.shape[-1])
        _, topk_indices = torch.topk(scores, k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_indices, 1)
        return scores * mask


class SparseAutoencoder(nn.Module):
    """
    Sparse linear autoencoder with TopK activation and truly tied weights.

    Config keys used:
      - d_in (int): input dimension
      - d_hidden (int): hidden (code) dimension
      - k (int): number of active units in TopK
      - allow_negative_topk (bool, optional): if False, negatives are zeroed before TopK
    """
    def __init__(self, config):
        super().__init__()

        d_in = config.get("d_in")
        d_hidden = config.get("d_hidden")
        k = config.get("k")
        allow_neg = bool(config.get("allow_negative_topk", False))

        # Basic validation to fail fast with clear messages
        if not isinstance(d_in, int) or not isinstance(d_hidden, int) or not isinstance(k, int):
            raise ValueError("SparseAutoencoder requires integer d_in, d_hidden, and k")
        if not (1 <= k <= d_hidden):
            raise ValueError(f"k must satisfy 1 <= k <= d_hidden ({d_hidden}), got k={k}")

        self.encoder = nn.Linear(d_in, d_hidden, bias=True)
        self.topk_activation = TopKActivation(k, allow_negative_topk=allow_neg)
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
