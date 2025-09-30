import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from lightning.pytorch.loggers import WandbLogger

from script.model.model_factory import get_encoder, infer_encoder_out_dim


class TopKActivation(nn.Module):
    """
    Activation layer that retains only the top-k values along the last dimension.

    If `allow_negative_topk` is False, the selection and outputs are based on
    ReLU(x), i.e., negative activations are zeroed and never selected.

    If `by_magnitude` is True, selection is by |x| but the original sign is kept.
    """
    def __init__(self, k: int, allow_negative_topk: bool = False, by_magnitude: bool = False):
        super().__init__()
        self.k = k
        self.allow_negative_topk = allow_negative_topk
        self.by_magnitude = by_magnitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.by_magnitude:
            # Use |x| to pick strongest units regardless of sign
            scores = x.abs()
            # Handle small dimensions gracefully
            k = min(self.k, scores.shape[-1])
            _, topk_indices = torch.topk(scores, k, dim=-1)
            mask = torch.zeros_like(scores)
            mask.scatter_(-1, topk_indices, 1)
            return x * mask
        else:
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
        by_mag = bool(config.get("topk_by_magnitude", False))  # Select by |x| when negatives carry signal

        # Basic validation to fail fast with clear messages
        if not isinstance(d_in, int) or not isinstance(d_hidden, int) or not isinstance(k, int):
            raise ValueError("SparseAutoencoder requires integer d_in, d_hidden, and k")
        if not (1 <= k <= d_hidden):
            raise ValueError(f"k must satisfy 1 <= k <= d_hidden ({d_hidden}), got k={k}")

        self.encoder = nn.Linear(d_in, d_hidden, bias=True)
        self.topk_activation = TopKActivation(k, allow_negative_topk=allow_neg, by_magnitude=by_mag)
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
    A minimal Lightning wrapper for the SparseAutoencoder model.
    """
    def __init__(self, config, encoder: nn.Module):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        self.config = config
        self.sae = SparseAutoencoder(config)
        self.loss_fn = nn.MSELoss()
        self.encoder = encoder
        self.encoder.freeze()

    def forward(self, x):
        return self.sae(x)

    def _step(self, batch: tuple | list):
        # Assuming batch is a tuple/list where the first element is the input tensor
        x = batch[0]

        # If an encoder is present, use it to get embeddings
        if self.encoder:
            with torch.no_grad():
                x = self.encoder(x)

        reconstruction = self(x)
        loss = self.loss_fn(reconstruction, x)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss_sae', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log(f'val_{self.loss_fn.__class__.__name__}_sae', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        lr = self.config.get("lr", 1e-3)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        if self.trainer.estimated_stepping_batches is None:
             # handle case where trainer is not available, e.g. during testing
            return optimizer

        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=self.trainer.estimated_stepping_batches
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
