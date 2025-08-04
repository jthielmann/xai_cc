import torch
import lightning as L
import torch.nn as nn
import torchvision
import torchmetrics
import torch.optim as optim
import wandb
from script.data_processing.image_transforms import get_transforms
from script.model.model_factory import get_encoder, infer_encoder_out_dim
import os
from script.data_processing.process_csv import generate_results_patient_from_loader
from script.train.generate_plots import generate_hists_2
import matplotlib.pyplot as plt
from script.data_processing.data_loader import get_dataset, load_best_smoothing, load_gene_weights
from torch.utils.data import DataLoader
import random
import sys
from io import BytesIO
from PIL import Image
from torch.optim.lr_scheduler import OneCycleLR, ChainedScheduler
sys.path.insert(0, '..')
from script.model.loss_functions import MultiGeneWeightedMSE
from script.model.lit_ae import SparseAutoencoder
from typing import Dict, Any


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

def get_model(config):
    return GeneExpressionRegressor(config)

def get_encoder_output_dim(encoder_type):
    if encoder_type == "dino":
        encoder_out_dim = 2048
    elif encoder_type == "resnet50random" or encoder_type == "resnet50imagenet":
        encoder_out_dim = 1000
    elif encoder_type == "unimodel":
        encoder_out_dim = 1536
    else:
        raise ValueError("encoder not found")
    return encoder_out_dim

class GeneExpressionRegressor(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        default_lr = self.config.get("head_lr", 1e-3)
        self.lrs = None
        self.encoder = get_encoder(self.config["encoder_type"])
        self.freeze_encoder = self.config['freeze_encoder']
        if self.freeze_encoder:
            for p in self.encoder.parameters(): p.requires_grad = False
        out_dim_encoder = infer_encoder_out_dim(self.encoder)
        for gene in self.config['genes']:
            relu_type = nn.LeakyReLU if self.config.get('use_leaky_relu') else nn.ReLU
            relu_instance = relu_type()

            layer = (
                nn.Sequential(relu_instance, nn.Linear(get_encoder_output_dim(self.config['encoder_type']), 1))
                if self.config['one_linear_out_layer']
                else nn.Sequential(
                    nn.Linear(out_dim_encoder, self.config['middle_layer_features']),
                    relu_instance,
                    nn.Linear(self.config['middle_layer_features'], 1),
                )
            )
            setattr(self, gene, layer)
        if config.get("sae", False):
            pos = config.get("sae_position", "pre")
            if pos == "pre":
                # SAE on the *features* coming *out* of the CNN
                d_in     = config["encoder_out_dim"]
                d_hidden = config["sae_hidden_dim"]
                k        = config["sae_k"]
                self.sae_pre  = SparseAutoencoder(d_in, d_hidden, k)
                self.sae_post = None
            elif pos == "post":
                # SAE on the *final head outputs*
                d_in     = len(config["genes"])
                d_hidden = config["sae_hidden_dim"]
                k        = config["sae_k"]
                self.sae_pre  = None
                self.sae_post = SparseAutoencoder(d_in, d_hidden, k)
            else:
                raise ValueError("sae_position must be 'pre' or 'post'")
        else:
            self.sae_pre  = None
            self.sae_post = None


        self.genes = self.config['genes']

        for g in self.genes:
            setattr(self, f"{g}_lr", default_lr)
        self.pearson = torchmetrics.PearsonCorrCoef(num_outputs=len(self.genes))
        if self.config["loss_fn_switch"] == "MSE":
            self.loss_fn = nn.MSELoss()
        elif self.config["loss_fn_switch"] == "WMSE" or self.config["loss_fn_switch"] == "weighted MSE":
            weight_dir = self.config["lds_weight_csv"]
            weights = load_gene_weights(weight_dir, self.genes)
            self.loss_fn = MultiGeneWeightedMSE(weights)

        self.num_training_batches = 0
        self.current_loss = torch.tensor(0.).to(self.device)
        self.best_loss = torch.tensor(float('inf')).to(self.device)
        self.is_online = self.config.get('log_to_wandb')

        if self.config.get('generate_scatters', False):
            self.table = wandb.Table(columns=["epoch","gene","lr","bins","scatter_plot"])
        if self.is_online:
            wandb.watch(self, log=None)
        self.encoder_lr = self.config.get("encoder_lr", 1e-3)  # encoder
        default_head_lr = self.config.get("head_lr", 1e-3)  # heads
        for g in self.config["genes"]:
            setattr(self, f"{g}_lr", default_head_lr)

    def forward(self, x):
        B = x.size(0)

        z = self.encoder(x)  # z: (B, encoder_out_dim)

        if self.sae_pre:
            z = self.sae_pre(z)

        outs = [getattr(self, g)(z) for g in self.genes]
        out  = outs[0] if len(outs)==1 else torch.cat(outs, dim=1)

        if self.sae_post:
            out = self.sae_post(out)

        return out

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch)
        self.log('train_' + self.config['loss_fn_switch'], loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._step(batch)
        self.log('val_' + self.config['loss_fn_switch'], loss, on_epoch=True)

        self.y_hats.append(y_hat)
        self.ys.append(y)
        self.current_loss += loss
        return loss

    def _step(self, batch):
        x,y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat, y

    def configure_optimizers(self):
        groups = []
        if not self.freeze_encoder:
            groups.append({
                "params": self.encoder.parameters(),
                "lr": self.encoder_lr
            })
        for g in self.genes:
            groups.append({
                "params": getattr(self, g).parameters(),
                "lr": getattr(self, f"{g}_lr")
            })
        opt = torch.optim.AdamW(groups)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=[pg["lr"] for pg in groups],
            total_steps=self.trainer.estimated_stepping_batches
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def on_fit_start(self):          # ← runs before the first batch
        os.makedirs(self.config["out_path"], exist_ok=True)

    def on_validation_epoch_start(self):
        self.current_loss = 0.0
        self.y_hats = []
        self.ys = []
        self.pearson.reset()

    def on_validation_epoch_end(self):
        # Skip sanity check epoch
        if not hasattr(self, 'sanity_skipped'):
            self.sanity_skipped = True
            return

        if not self.config.get('debug'):
            torch.save(self.state_dict(), os.path.join(self.config['out_path'], "latest.pth"))
            if self.current_loss < self.best_loss:
                self.best_loss = self.current_loss
                # Save the best model
                best_model_path = os.path.join(self.config['out_path'], "best_model.pth")
                torch.save(self.state_dict(), best_model_path)

                if self.is_online:
                    wandb.run.summary["best_val_loss"] = self.best_loss
                    wandb.run.summary["best_val_epoch"] = self.current_epoch
                    #wandb.save(best_model_path, base_path=self.config['out_path'])
                    wandb.log({"epoch": self.current_epoch})
        # Aggregate outputs
        y_hat = torch.cat(self.y_hats, dim=0)
        y_true = torch.cat(self.ys, dim=0)
        pearson = self.pearson(y_hat, y_true)
        if len(self.genes) > 1:
            pearson_dict = {f"pearson_{g}": pearson[i] for i, g in enumerate(self.genes)}
        else: # one gene
            pearson_dict = {f"pearson_{self.genes[0]}": pearson}
        if self.is_online:
            self.log_dict(pearson_dict, on_epoch=True)

        # Generate scatter plots if requested
        if self.config.get('generate_scatters', False):
            model_name = f"ep_{self.current_epoch}"
            results_file = os.path.join(self.config['out_path'], "_" + str(random.randbytes(4).hex()) + "_results.csv")

            # iterate patients
            for patient in self.config['val_samples']:
                val_ds = get_dataset(
                    self.config['data_dir'],
                    genes=self.genes,
                    samples=[patient],
                    transforms=get_transforms(self.config),
                    bins=self.config.get("bins", 0),
                    gene_data_filename=self.config['gene_data_filename'],
                    max_len=100 if self.config.get('debug') else None
                )
                loader = DataLoader(
                    val_ds,
                    batch_size=self.config['batch_size'],
                    shuffle=False,
                    num_workers=self.config.get('num_workers', 0),
                    pin_memory=False
                )
                generate_results_patient_from_loader(self, loader, results_file, patient)


            figures = generate_hists_2(self, results_file, out_file_appendix="_" + model_name)
            for gene, fig in figures.items():
                buf = BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                img = Image.open(buf)
                if self.is_online:
                    self.table.add_data(
                        self.current_epoch,
                        gene,
                        self.config['learning_rate'],
                        self.config.get("bins", 0),
                        wandb.Image(img, caption=gene)
                    )
                plt.close(fig)
            os.remove(results_file)


    def on_train_epoch_end(self):
        if not self.config.get('debug'):
            torch.save(self.state_dict(), self.config["out_path"] + "/latest.pth")

    def on_train_end(self):
        if self.is_online:
            if hasattr(self, 'table'):
                wandb.log({'scatter_table': self.table})

    # to update after lr tuning
    def update_lr(self, lrs):
        self.lrs = lrs
        if not self.freeze_encoder:
            self.encoder_lr = lrs["encoder"]  # encoder
        for g in self.config["genes"]:
            setattr(self, f"{g}_lr", lrs[g])

    def get_lr(self, param_name):
        return self.lrs[param_name]
