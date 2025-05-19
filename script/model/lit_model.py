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

        # Unpack hyperparameters
        self.config = config
        self.encoder = get_encoder(self.config["encoder_type"])
        self.freeze_pretrained = self.config['freeze_pretrained']
        if self.freeze_pretrained:
            for p in self.encoder.parameters(): p.requires_grad = False

        # Build gene heads
        for gene in self.config['genes']:
            setattr(self, gene,
                    nn.Sequential(nn.Linear(self.config['pretrained_out_dim'], self.config['middle_layer_features']),
                                  nn.ReLU(),
                                  nn.Linear(self.config['middle_layer_features'], 1)))
        self.genes = self.config['genes']

        # Metrics and loss
        self.pearson = torchmetrics.PearsonCorrCoef(num_outputs=len(self.genes))
        self.loss_fn = nn.MSELoss()

        # Training bookkeeping
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
        # accumulate for epoch end
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

        # --- Backbone & Projection Heads ---
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

        # --- Loss & Momentum Scheduling ---
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

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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
        # Save only the latest checkpoint each epoch
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
    def __init__(self, encoder, encoder_out_features, middle_layer_features, ae_out_features, epochs, learning_rate, logging, out_path, generate_scatters):
        super().__init__()
        self.encoder = encoder
        self.encoder_out_features = encoder_out_features
        self.ae_out_features = ae_out_features
        self.epochs = epochs
        # model setup
        for param in self.encoder.parameters():
            param.requires_grad = False

        # metrics
        self.loss = torch.nn.MSELoss()
        self.optimizer = None
        self.scheduler = None
        self.current_loss = torch.Tensor([0])
        self.best_loss = torch.Tensor([float("Inf")])
        self.out_path = out_path

        self.y_hats = []
        self.ys = []
        self.val_epoch_counter = 0
        self.best_val_epoch = 0
        self.num_training_batches = 0
        self.learning_rate = learning_rate
        self.logging = logging
        if logging:
            self.save_hyperparameters(ignore=['encoder'])
        self.is_mps_available = torch.backends.mps.is_available()

        self.sane = False # skip things for sanity check
        self.generate_scatters = generate_scatters

        self.layers = nn.Sequential(nn.Linear(encoder_out_features, middle_layer_features), nn.ReLU(),
                                    nn.Linear(middle_layer_features, ae_out_features))

    def forward(self, x):
        x = self.encoder(x)
        x = self.layers(x)
        return x

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.common_step(batch, batch_idx)
        if self.log_loss:
            self.log_dict({"training " + self.error_metric_name: loss}, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.common_step(batch, batch_idx)
        # validation loss must be logged so that we can use early stopping with wandb
        self.log_dict({"validation " + self.error_metric_name: loss}, on_step=False, on_epoch=True)
        self.current_loss += loss

        self.y_hats.append(y_hat)
        self.ys.append(y)
        return loss

    def common_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        if torch.any(loss.isnan()):
            print("loss is nan")
            exit(1)
        return loss, y_hat, y

    def configure_optimizers(self):
        params = []
        params.append({"params": self.encoder.parameters(), "lr": self.learning_rate})
        for gene in self.genes:
            params.append({"params": getattr(self, gene).parameters(), "lr": self.learning_rate})
        optimizer = optim.AdamW(params)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, epochs=self.epochs, steps_per_epoch=self.num_training_batches)
        self.scheduler = scheduler
        self.optimizer = optimizer
        return {"lr_scheduler": scheduler, "optimizer": optimizer}

    def on_validation_start(self) -> None:
        # in the init function self.device is cpu, so we can do it e.g. here
        self.best_loss = self.best_loss.to(self.device)

    def on_validation_epoch_start(self):
        self.current_loss = torch.Tensor([0]).to(self.device)
        self.y_hats = []
        self.ys = []

    def on_validation_epoch_end(self):
        if not self.sane:
            self.sane = True
            return # skip the first validation epoch from sanity check
        self.y_hats = torch.cat(self.y_hats, dim=0)
        self.ys = torch.cat(self.ys, dim=0)
        pearsons = self.pearson(self.y_hats, self.ys)
        self.pearson_values.append(pearsons)
        self.val_epoch_counter += 1
        if self.logging:
            if len(self.genes) == 1:
                self.log({"pearsons_" + self.genes[0]: pearsons})
            else:
                for gene in self.genes:
                    self.log({"pearsons_" + gene: pearsons[self.genes.index(gene)]})
        if lit_config["debug"]:
            torch.save(self.state_dict(), self.out_path + "/" + str(self.current_epoch) + ".pth")
        if len(self.pearson_values) == 1 or (torch.abs(self.pearson_values[-1]) > max(torch.abs(torch.stack(self.pearson_values[:-1])))).item():
            self.best_loss = self.current_loss
            torch.save(self.state_dict(), self.out_path + "/best_model.pth")
            self.best_val_epoch = self.val_epoch_counter
            if self.logging:
                wandb.run.summary["best_val_loss"] = self.best_loss
                wandb.run.summary["best_val_loss_avg"] = self.best_loss / self.num_training_batches
                wandb.run.summary["best_val_epoch"] = self.best_val_epoch

        if self.generate_scatters:
            model_names = ["ep_" + str(self.current_epoch)]
            for model_name in model_names:
                results_file_name = self.out_path + "/results.csv"
                if os.path.exists(results_file_name):
                    os.remove(results_file_name)

                for patient in lit_config["val_samples"]:
                    val_dataset = get_dataset(lit_config["data_dir"], genes=self.genes, samples=[patient],
                                              transforms=get_transforms(), bins=wandb.run.config["bins"],
                                              gene_data_filename=lit_config["gene_data_filename_val"])
                    loader = DataLoader(val_dataset, batch_size=lit_config["batch_size"], shuffle=False,
                               num_workers=lit_config["num_workers"], pin_memory=False)
                    generate_results_patient_from_loader(self, loader, results_file_name, patient)
                figure_paths = generate_hists_2(self, results_file_name, out_file_appendix="_" + model_name)
                wandb.log({"hist " + str(self.current_epoch) : [wandb.Image(path) for path in figure_paths]})
                for path in figure_paths:
                    plt.imshow(plt.imread(path))
                    plt.show()


    def on_train_epoch_end(self):
        torch.save(self.state_dict(), self.out_path + "/latest.pth")

    def on_train_end(self):
        print("device used", self.device)

        best_pearsons = []
        best_pearson_epochs = []
        for i in range(len(self.genes)):
            if len(self.genes) == 1:
                best_epoch = torch.argmax(torch.stack(self.pearson_values).abs(), dim=0).item()
                best_pearson_values = round(self.pearson_values[best_epoch].item(), 3)
            else:
                best_epoch = torch.argmax(torch.stack(self.pearson_values)[:,i].abs(), dim=0).item()
                best_pearson_values = round(self.pearson_values[best_epoch][i].item(), 3)

            best_pearson_epochs.append(best_epoch)
            best_pearsons.append(best_pearson_values)
        best_pearson_dict = {self.genes[i]: best_pearsons[i] for i in range(len(self.genes))}
        print("pearsons", self.pearson_values)

        print("best_pearson_dict", best_pearson_dict)
        best_pearson_epoch_dict = {self.genes[i]: best_pearson_epochs[i] for i in range(len(self.genes))}
        print("best_pearson_epoch_dict", best_pearson_epoch_dict)
        if not self.logging:
            return
        wandb.run.summary["best_pearsons"] = best_pearson_dict
        wandb.run.summary["best_pearson_epochs"] = best_pearson_epoch_dict
        if lit_config["debug"]:
            print("best_pearsons", best_pearson_dict)
            print("best_pearson_epochs", best_pearson_epoch_dict)
            print("pearsons", self.pearson_values)

        if self.generate_scatters:
            model_names = ["best_model.pth"]
            for model_name in model_names:
                results_file_name = self.out_path + "/results.csv"
                if os.path.exists(results_file_name):
                    os.remove(results_file_name)

                for patient in lit_config["val_samples"]:
                    val_dataset = get_dataset(lit_config["data_dir"], genes=self.genes, samples=[patient],
                                                   transforms=get_transforms(), bins=wandb.run.config["bins"])
                    loader = DataLoader(val_dataset, batch_size=lit_config["batch_size"], shuffle=False,
                               num_workers=lit_config["num_workers"], pin_memory=False)
                    generate_results_patient_from_loader(self, loader, results_file_name, patient)
                figure_paths = generate_hists_2(self, results_file_name, out_file_appendix="_" + model_name)
                self.log({"hist": [wandb.Image(path) for path in figure_paths]})
                for path in figure_paths:
                    plt.imshow(plt.imread(path))
                    plt.show()
    def set_num_training_batches(self, num_batches):
        self.num_training_batches = num_batches