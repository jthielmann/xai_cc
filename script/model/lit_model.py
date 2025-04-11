import torch
import lightning as L
import torch.nn as nn
import torchvision
import torchmetrics
import torch.optim as optim
import wandb
from script.data_processing.image_transforms import get_transforms
from script.configs.lit_config import get_encoder
from script.configs.lit_config import lit_config
import os
from script.data_processing.process_csv import generate_results_patient_from_loader
from script.train.generate_plots import generate_hists_2
import matplotlib.pyplot as plt
from script.data_processing.data_loader import get_dataset
from torch.utils.data import DataLoader
import script.configs.dino_config as dino_config

import copy

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
import sys
from io import BytesIO
from PIL import Image

sys.path.insert(0, '..')

def load_model(path, config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


    encoder = get_encoder(config["encoder_type"])
    if "pretrained_out_dim" not in config:
        config["pretrained_out_dim"] = 1000
    if "middel_layer_features" not in config:
        config["middel_layer_features"] = 64
    model = LightiningNN(encoder=encoder, genes=config["genes"], pretrained_out_dim=config["pretrained_out_dim"], middel_layer_features=config["middel_layer_features"], out_path=path, error_metric_name=config["error_metric_name"], freeze_pretrained=config["freeze_pretrained"], epochs=config["epochs"], learning_rate=config["learning_rate"], use_transforms=config["use_transforms_in_model"], logging=False)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model

# dummy default values to enable easy dummy building for loading from pth
class LightiningNN(L.LightningModule):
    def __init__(self, genes, encoder, pretrained_out_dim, middel_layer_features, out_path, error_metric_name,
                 freeze_pretrained, epochs, learning_rate, bins, use_transforms, logging, log_loss=False,
                 generate_scatters=True, ae=None):
        super().__init__()

        self.epochs = epochs
        # model setup
        self.encoder = encoder
        self.freeze_pretrained = freeze_pretrained
        if self.freeze_pretrained:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if genes is None:
            genes = []
        for gene in genes:
            setattr(self, gene, nn.Sequential(nn.Linear(pretrained_out_dim, middel_layer_features), nn.ReLU(),
                                              nn.Linear(middel_layer_features, 1)))
        self.genes = genes
        # metrics
        self.pearson = torchmetrics.PearsonCorrCoef(num_outputs=len(genes))
        self.loss = torch.nn.MSELoss()
        self.optimizer = None
        self.scheduler = None
        self.error_metric_name = error_metric_name
        self.current_loss = torch.Tensor([0])
        self.best_loss = torch.Tensor([float("Inf")])
        self.out_path = out_path
        self.pearson_values = []
        self.y_hats = []
        self.ys = []
        self.val_epoch_counter = 0
        self.best_val_epoch = 0
        self.num_training_batches = 0
        self.learning_rate = learning_rate
        self.bins = bins
        self.logging = logging
        self.log_loss = log_loss
        if logging:
            self.save_hyperparameters(ignore=['encoder'])
        self.is_mps_available = torch.backends.mps.is_available()
        if use_transforms:
            self.transforms = get_transforms()
        else:
            self.transforms = None
        self.sane = False # skip things for sanity check
        self.generate_scatters = generate_scatters
        self.ae = ae
        self.table = wandb.Table(columns=["epoch", "gene", "learning rate", "bins", "scatter_plot"])

    def forward(self, x):
        if self.transforms:
            # some functions are not supported on mps device as of jan 2025, use the transforms on cpu via the datamodule
            # note: this slows down the training process for mps when using transforms because copy but it is what it is
            if self.is_mps_available:
                x = x.cpu()
            x = self.transforms(x)
            if self.is_mps_available:
                x = x.to(self.device)
                x = x.to(torch.float32)
        x = self.encoder(x)
        if self.ae:
            x = self.ae(x)
        out = []
        for gene in self.genes:
            out.append(getattr(self, gene)(x))
        if len(out) == 1:
            return out[0]
        return torch.cat(out, dim=1)

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
        seen = set()

        def add_unique_params(module):
            for p in module.parameters():
                if p not in seen:
                    seen.add(p)
                    yield p

        params.append({"params": list(add_unique_params(self.encoder)), "lr": self.learning_rate})
        for gene in self.genes:
            head = getattr(self, gene)
            params.append({"params": list(add_unique_params(head)), "lr": self.learning_rate})
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
                self.log("pearsons_" + self.genes[0], pearsons)
            else:
                for i, gene in enumerate(self.genes):
                    self.log("pearsons_" + gene, pearsons[i])
        if lit_config["debug"]:
            torch.save(self.state_dict(), self.out_path + "/" + str(self.current_epoch) + ".pth")
        if len(self.genes) == 1:
            if len(self.pearson_values) == 1 or (torch.abs(self.pearson_values[-1]) > max(torch.abs(torch.stack(self.pearson_values[:-1])))).item():
                self.best_loss = self.current_loss
                torch.save(self.state_dict(), self.out_path + "/best_model.pth")
                self.best_val_epoch = self.val_epoch_counter
                if self.logging:
                    wandb.run.summary["best_val_loss"] = self.best_loss
                    wandb.run.summary["best_val_loss_avg"] = self.best_loss / self.num_training_batches
                    wandb.run.summary["best_val_epoch"] = self.best_val_epoch
        else:
            if len(self.pearson_values) == 1 or torch.sum(torch.abs(self.pearson_values[-1]) - torch.abs(torch.stack(self.pearson_values[:-1]))).item() > 0:
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
                                                   gene_data_filename=lit_config["gene_data_filename_val"], max_len=100 if lit_config["debug"] else None)
                    loader = DataLoader(val_dataset, batch_size=lit_config["batch_size"], shuffle=False,
                               num_workers=lit_config["num_workers"], pin_memory=False)
                    generate_results_patient_from_loader(self, loader, results_file_name, patient)
                figures = generate_hists_2(self, results_file_name, out_file_appendix="_" + model_name)

                for gene, fig in figures.items():
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    img = Image.open(buf)
                    self.table.add_data(self.current_epoch, gene, self.learning_rate, self.bins, wandb.Image(img, caption=gene))
                    plt.close(fig)


    def on_train_epoch_end(self):
        torch.save(self.state_dict(), self.out_path + "/latest.pth")

    def on_train_end(self):
        print("device used", self.device)
        wandb.log({"scatter_table": self.table}, step=self.current_epoch)

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

    def set_num_training_batches(self, num_batches):
        self.num_training_batches = num_batches


# adapted from https://docs.lightly.ai/self-supervised-learning/examples/dino.html#dino
class DINO(L.LightningModule):
    def __init__(self):
        super().__init__()
        backbone = dino_config.backbone
        if backbone == "resnet18":
            resnet = torchvision.models.resnet18()
            input_dim = 512
        else:#if backbone == "resnet50":
            resnet = torchvision.models.resnet50()
            input_dim = 2048
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        # instead of a resnet you can also use a vision transformer backbone as in the
        # original paper (you might have to reduce the batch size in this case):
        # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
        # input_dim = backbone.embed_dim

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def common_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim


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