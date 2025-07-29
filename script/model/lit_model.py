import os
from io import BytesIO
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import torchmetrics
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import wandb
from PIL import Image

from script.data_processing.image_transforms import get_transforms
from script.data_processing.process_csv import generate_results_patient_from_loader
from script.train.generate_plots import generate_hists_2
from script.data_processing.data_loader import get_dataset, load_gene_weights
from script.model.model_factory import get_encoder
from script.model.loss_functions import MultiGeneWeightedMSE
from lit_ae import SparseAutoencoder
import matplotlib.pyplot as plt


def load_model(path: str, config: Dict[str, Any]) -> L.LightningModule:
    """
    Load a encoder model checkpoint into a LightningModule for inference.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = GeneExpressionRegressor(config)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def get_model(config: Dict[str, Any]) -> L.LightningModule:
    """
    Instantiate a new supervised Lightning model.
    """
    return GeneExpressionRegressor(config)


class GeneExpressionRegressor(L.LightningModule):
    """
    Multi-gene regression using a encoder encoder with optional sparse autoencoders.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        # Automatically save all hyperparameters in `self.hparams.config`
        self.save_hyperparameters('config')
        cfg = self.hparams.config

        # Encoder
        self.encoder = get_encoder(cfg['encoder_type'])
        if cfg.get('freeze_encoder', False):
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Gene-specific heads
        self.heads = nn.ModuleDict({
            gene: nn.Sequential(
                nn.Linear(cfg['encoder_out_dim'], cfg['middle_layer_features']),
                nn.LeakyReLU() if cfg.get('use_leaky_relu') else nn.ReLU(),
                nn.Linear(cfg['middle_layer_features'], 1)
            )
            for gene in cfg['genes']
        })

        # Optional sparse autoencoder
        self.sae_pre: SparseAutoencoder | None = None
        self.sae_post: SparseAutoencoder | None = None
        if cfg.get('sae', False):
            pos = cfg.get('sae_position', 'pre')
            if pos == 'pre':
                self.sae_pre = SparseAutoencoder(
                    cfg['encoder_out_dim'], cfg['sae_hidden_dim'], cfg['sae_k']
                )
            elif pos == 'post':
                self.sae_post = SparseAutoencoder(
                    len(cfg['genes']), cfg['sae_hidden_dim'], cfg['sae_k']
                )
            else:
                raise ValueError("sae_position must be 'pre' or 'post'")

        # Loss function
        if cfg['loss_fn_switch'] == 'MSE':
            self.loss_fn = nn.MSELoss()
        else:
            weights = load_gene_weights(cfg['lds_weight_csv'], cfg['genes'])
            self.loss_fn = MultiGeneWeightedMSE(weights)

        # Metrics
        self.pearson = torchmetrics.PearsonCorrCoef(num_outputs=len(cfg['genes']))

        # Best validation loss (registered buffer so it's moved with the model)
        self.register_buffer('best_loss', torch.tensor(float('inf')))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self.sae_pre:
            z = self.sae_pre(z)
        outs = [self.heads[gene](z) for gene in self.hparams.config['genes']]
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
        if self.sae_post:
            out = self.sae_post(out)
        return out

    def _shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        loss, _, _ = self._shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> None:
        loss, y_hat, y = self._shared_step(batch)
        self.log('val_loss', loss, on_epoch=True)
        # Accumulate for epoch-end metrics
        self.y_hats.append(y_hat)
        self.ys.append(y)
        self.current_loss += loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        # Skip initial sanity check
        if not hasattr(self, 'sanity_skipped'):
            self.sanity_skipped = True
        else:
            # Compute and log Pearson per gene
            y_hat = torch.cat(self.y_hats)
            y_true = torch.cat(self.ys)
            pearson_vals = self.pearson(y_hat, y_true)
            for i, gene in enumerate(self.hparams.config['genes']):
                self.log(f'pearson_{gene}', pearson_vals[i])

            # Checkpointing
            out_path = self.hparams.config['out_path']
            os.makedirs(out_path, exist_ok=True)
            torch.save(self.state_dict(), os.path.join(out_path, 'latest.pt'))
            if self.current_loss < self.best_loss:
                self.best_loss = self.current_loss
                torch.save(self.state_dict(), os.path.join(out_path, 'best_model.pt'))

        # Reset for next epoch
        self.y_hats = []
        self.ys = []
        self.current_loss = torch.tensor(0.0, device=self.device)

    def configure_optimizers(self) -> Dict[str, Any]:
        cfg = self.hparams.config
        optimizer = optim.AdamW(self.parameters(), lr=cfg['learning_rate'])
        scheduler = {
            'scheduler': OneCycleLR(
                optimizer,
                max_lr=cfg['learning_rate'],
                total_steps=self.trainer.estimated_stepping_batches
            ),
            'interval': 'step'
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def ScatterPlotCallback(config: Dict[str, Any]) -> L.Callback:
    """
    Factory for a callback that generates per-epoch scatter histograms logged to WandB.
    """
    class _ScatterPlotCallback(L.Callback):
        def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: GeneExpressionRegressor) -> None:
            table = wandb.Table(columns=["epoch", "gene", "hist_plot"])
            epoch = trainer.current_epoch
            for patient in config['val_samples']:
                ds = get_dataset(
                    config['data_dir'], genes=config['genes'], samples=[patient],
                    transforms=get_transforms(config), bins=config.get('bins', 0),
                    gene_data_filename=config['gene_data_filename']
                )
                loader = DataLoader(ds, batch_size=config['batch_size'], shuffle=False)
                tmp_csv = os.path.join(config['out_path'], f"_{epoch}_results.csv")
                generate_results_patient_from_loader(pl_module, loader, tmp_csv, patient)
                figs = generate_hists_2(pl_module, tmp_csv, out_file_appendix=f"_{epoch}")
                for gene, fig in figs.items():
                    buf = BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    img = Image.open(buf)
                    table.add_data(epoch, gene, wandb.Image(img, caption=gene))
                    plt.close(fig)
                os.remove(tmp_csv)
            wandb.log({'scatter_table': table})
    return _ScatterPlotCallback()
