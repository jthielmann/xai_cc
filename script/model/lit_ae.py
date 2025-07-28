import os

import torchmetrics
from torch import nn
import torch
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR

from script.model.model_factory import get_encoder
import lightning as L

class TopKActivation(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        _, topk_indices = torch.topk(x, self.k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, topk_indices, 1)
        return x * mask


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in, d_hidden, k):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_hidden, bias=True)
        self.topk_activation = TopKActivation(k)
        self.decoder = nn.Linear(d_hidden, d_in, bias=True)
        self.last_sparse = None
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T

    def forward(self, x):
        encoded = self.encoder(x)
        sparse = self.topk_activation(encoded)
        self.last_sparse = sparse.detach()
        decoded = self.decoder(sparse)
        return decoded

# adapted from https://docs.lightly.ai/self-supervised-learning/examples/dino.html#dino


class Autoencoder(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)          # ← logs to Lightning / W&B

        self.encoder = get_encoder(config["encoder_type"])
        if config.get("freeze_pretrained", False):
            for p in self.encoder.parameters():
                p.requires_grad = False

        in_dim  = config["pretrained_out_dim"]
        hid_dim = config.get("middle_layer_features", 256)
        out_dim = config["ae_out_features"]

        act = nn.LeakyReLU if config.get("use_leaky_relu", False) else nn.ReLU
        self.decoder = nn.Sequential(
            nn.Linear(in_dim,  hid_dim),
            act(),
            nn.Linear(hid_dim, out_dim)
        )

        # ───────────────────────────────────── training helpers
        self.loss_fn   = nn.MSELoss()
        self.pearson   = torchmetrics.PearsonCorrCoef(num_outputs=out_dim)
        self.best_loss = torch.tensor(float("inf"))

        # will be filled by the trainer before `configure_optimizers` is called
        self.num_training_batches = 0

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def _shared_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss  = self.loss_fn(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, _):
        loss, _, _ = self._shared_step(batch)
        self.log(f"train_{self.hparams.loss_fn_switch}", loss,
                 on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_fit_start(self):
        os.makedirs(self.config["out_path"], exist_ok=True)

    def validation_step(self, batch, _):
        loss, y_hat, y = self._shared_step(batch)
        self.log(f"val_{self.hparams.loss_fn_switch}", loss, on_epoch=True)
        self.val_yhats.append(y_hat)
        self.val_ys.append(y)
        self.val_loss += loss
        return loss

    def on_validation_epoch_start(self):
        self.val_yhats, self.val_ys = [], []
        self.val_loss = torch.tensor(0., device=self.device)

    def on_validation_epoch_end(self):
        # skip Lightning’s internal “sanity” epoch
        if not hasattr(self, "_sanity_done"):
            self._sanity_done = True
            return

        # aggregate metrics
        yh  = torch.cat(self.val_yhats)
        yt  = torch.cat(self.val_ys)
        r   = self.pearson(yh, yt)
        self.log("pearson", r)

        # checkpointing
        latest = os.path.join(self.hparams.out_path, "latest.pth")
        torch.save(self.state_dict(), latest)

        if self.val_loss < self.best_loss:
            self.best_loss = self.val_loss
            best = os.path.join(self.hparams.out_path, "best_model.pth")
            torch.save(self.state_dict(), best)

    def configure_optimizers(self):
        if self.num_training_batches == 0:
            raise RuntimeError(
                "`set_num_training_batches` must be called before configure_optimizers()"
            )

        params = [
            {"params": self.encoder.parameters(), "lr": self.hparams.learning_rate},
            {"params": self.decoder.parameters(), "lr": self.hparams.learning_rate},
        ]
        opt = optim.AdamW(params)
        sch = OneCycleLR(
            opt,
            max_lr=self.hparams.learning_rate,
            epochs=self.hparams.epochs,
            steps_per_epoch=self.num_training_batches,
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "interval": "step"}}

    def set_num_training_batches(self, n: int):
        self.num_training_batches = n