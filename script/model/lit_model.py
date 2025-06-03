import torch
import lightning as L
import torch.nn as nn
import torchvision
import torchmetrics
import torch.optim as optim
import wandb
from script.data_processing.image_transforms import get_transforms
from script.model.model_factory import get_encoder
import os
from script.data_processing.process_csv import generate_results_patient_from_loader
from script.train.generate_plots import generate_hists_2
import matplotlib.pyplot as plt
from script.data_processing.data_loader import get_dataset
from torch.utils.data import DataLoader
import random

import copy

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
import sys
from io import BytesIO
from PIL import Image
from torch.optim.lr_scheduler import OneCycleLR
sys.path.insert(0, '..')

def load_model(path, config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


    encoder = get_encoder(config["encoder_type"])
    if "pretrained_out_dim" not in config:
        config["pretrained_out_dim"] = 1000
    if "middle_layer_features" not in config:
        config["middle_layer_features"] = 64
    model = LightiningNN(config, encoder)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model

def get_model(config):
    return LightiningNN(config)

class LightiningNN(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.encoder = get_encoder(self.config["encoder_type"])
        self.freeze_pretrained = self.config['freeze_pretrained']
        if self.freeze_pretrained:
            for p in self.encoder.parameters(): p.requires_grad = False

        for gene in self.config['genes']:
            relu = nn.LeakyReLU if self.config['use_leaky_relu'] else nn.ReLU
            if self.config['one_linear_out_layer']:
                setattr(self, gene,
                        nn.Sequential(
                            relu,
                            nn.Linear(self.config['pretrained_out_dim'],1)))
            else:
                setattr(self, gene,
                        nn.Sequential(nn.Linear(self.config['pretrained_out_dim'], self.config['middle_layer_features']),
                                      relu,
                                      nn.Linear(self.config['middle_layer_features'], 1)))
        self.genes = self.config['genes']

        self.pearson = torchmetrics.PearsonCorrCoef(num_outputs=len(self.genes))
        self.loss_fn = nn.MSELoss()

        self.num_training_batches = 0
        self.current_loss = torch.tensor(0.).to(self.device)
        self.best_loss = torch.tensor(float('inf')).to(self.device)
        if self.config.get('generate_scatters', False):
            self.table = wandb.Table(columns=["epoch","gene","lr","bins","scatter_plot"])
        wandb.watch(self, log=None)

    def forward(self, x):
        x = self.encoder(x)
        outs = [getattr(self, g)(x) for g in self.genes]
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch)
        self.log('train_' + self.config['error_metric_name'], loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._step(batch)
        self.log('val_' + self.config['error_metric_name'], loss, on_epoch=True)

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
        params = [{'params': self.encoder.parameters(), 'lr': self.config['learning_rate']}]
        for gene in self.genes:
            params.append({'params': getattr(self, gene).parameters(), 'lr': self.config['learning_rate']})
        optimizer = optim.AdamW(params)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config['learning_rate'],
            epochs=self.config['epochs'],
            steps_per_epoch=self.num_training_batches)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

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
        torch.save(self.state_dict(), os.path.join(self.config['out_path'], "latest.pth"))
        if self.current_loss < self.best_loss:
            self.best_loss = self.current_loss
            wandb.run.summary["best_val_loss"] = self.best_loss
            wandb.run.summary["best_val_epoch"] = self.current_epoch
            # Save the best model
            best_model_path = os.path.join(self.config['out_path'], "best_model.pth")
            torch.save(self.state_dict(), best_model_path)
            wandb.save(best_model_path, base_path=self.config['out_path'])
            wandb.log({"epoch": self.current_epoch})
        # Aggregate outputs
        y_hat = torch.cat(self.y_hats, dim=0)
        y_true = torch.cat(self.ys, dim=0)
        pearson = self.pearson(y_hat, y_true)
        self.log(f"pearson_{self.genes[0] if len(self.genes)==1 else 'all'}", pearson)

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
                    bins=self.config['bins'],
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
                self.table.add_data(
                    self.current_epoch,
                    gene,
                    self.config['learning_rate'],
                    self.config['bins'],
                    wandb.Image(img, caption=gene)
                )
                plt.close(fig)
            os.remove(results_file)


    def on_train_epoch_end(self):
        torch.save(self.state_dict(), self.config["out_path"] + "/latest.pth")

    def on_train_end(self):
        if hasattr(self, 'table'):
            wandb.log({'scatter_table': self.table})


    def set_num_training_batches(self, num_batches):
        self.num_training_batches = num_batches


# adapted from https://docs.lightly.ai/self-supervised-learning/examples/dino.html#dino
class DINO(L.LightningModule):
    def __init__(self, dino_config, lr=0.001):
        super().__init__()
        self.config = dino_config
        self.lr = lr

        backbone_name = self.config['encoder_type']
        if backbone_name == "resnet18":
            resnet = torchvision.models.resnet18(pretrained=False)
            input_dim = 512
        elif backbone_name == "resnet50":
            resnet = torchvision.models.resnet50(pretrained=False)
            input_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone {backbone_name}")

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
        # Cosine momentum schedule over full epochs
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
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        print("Configuring optimizers")
        # Ensure num_training_batches is set
        if not hasattr(self, 'num_training_batches') or self.num_training_batches <= 0:
            raise ValueError(
                "`num_training_batches` must be set (via `set_num_training_batches()`) before configuring optimizers"
            )
        wd = self.config.get("weight_decay", 0.0)
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
                'interval': 'step',  # update per batch
                'frequency': 1
            }
        }

    def on_train_epoch_end(self):
        latest_path = os.path.join(self.config['out_path'], 'latest.pth')
        torch.save(self.state_dict(), latest_path)

    def on_validation_start(self):
        self.current_loss = torch.tensor(0.).to(self.device)

    def on_validation_end(self):
        # Always save latest after validation
        latest_path = os.path.join(self.config['out_path'], 'latest.pth')
        torch.save(self.state_dict(), latest_path)

        # Save and log best model only if improved
        if self.current_loss < self.best_loss:
            self.best_loss = self.current_loss
            wandb.run.summary['best_val_loss'] = self.best_loss
            wandb.run.summary['best_val_epoch'] = self.current_epoch
            best_path = os.path.join(self.config['out_path'], 'best_model.pth')
            torch.save(self.state_dict(), best_path)
            wandb.log({'epoch': self.current_epoch})

    def set_num_training_batches(self, num_batches):
        self.num_training_batches = num_batches


class Autoencoder(L.LightningModule):
    def __init__(self, config):
        """
        Expected `config` fields (mirrors the ones used elsewhere):

        ─ encoder_type            (str)   – name understood by `get_encoder`
        ─ pretrained_out_dim      (int)   – feature dim at encoder output
        ─ middle_layer_features   (int)   – hidden size in the lightweight head
        ─ ae_out_features         (int)   – dimensionality of the regression target
        ─ learning_rate           (float) – peak LR for One-Cycle
        ─ epochs                  (int)   – total epochs (needed for One-Cycle)
        ─ freeze_pretrained       (bool)  – freeze the CNN backbone?
        ─ out_path                (str)   – where checkpoints are written
        ─ error_metric_name       (str)   – only for the W&B chart title
        """
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
        self.log(f"train_{self.hparams.error_metric_name}", loss,
                 on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, y_hat, y = self._shared_step(batch)
        self.log(f"val_{self.hparams.error_metric_name}", loss, on_epoch=True)
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