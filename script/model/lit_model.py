import torch
import lightning as L
import torch.nn as nn
import torchmetrics
import wandb
import torch.optim as optim
from script.data_processing.image_transforms import get_transforms
from script.configs.lit_config import get_encoder
# lightning module

def load_model(path, config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


    encoder = get_encoder(config["encoder_type"])
    if "pretrained_out_dim" not in config:
        config["pretrained_out_dim"] = 1000
    if "middel_layer_features" not in config:
        config["middel_layer_features"] = 64
    model = LightiningNN(encoder=encoder, genes=config["genes"], pretrained_out_dim=config["pretrained_out_dim"], middel_layer_features=config["middel_layer_features"], out_path=path, error_metric_name=config["error_metric_name"], freeze_pretrained=config["freeze_pretrained"], epochs=config["epochs"], learning_rate=config["learning_rate"], use_transforms=config["use_transforms_in_model"], logging=False)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

# dummy default values to enable easy dummy building for loading from pth
class LightiningNN(L.LightningModule):
    def __init__(self, genes, encoder, pretrained_out_dim, middel_layer_features, out_path, error_metric_name,
                 freeze_pretrained, epochs, learning_rate, use_transforms, logging):
        super(LightiningNN, self).__init__()

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
        self.logging = logging
        if logging:
            self.save_hyperparameters(ignore=['encoder'])
        self.is_mps_available = torch.backends.mps.is_available()
        if use_transforms:
            self.transforms = get_transforms()
        else:
            self.transforms = None
        self.sane = False # skip things for sanity check

    def forward(self, x):
        if self.transforms:
            # some functions are not supported on mps device as of jan 2025, use the transforms on cpu via the datamodule
            # note: this slows down the training process for mps when using transforms because copy but it is what it is
            if self.is_mps_available:
                x = x.cpu()
            x = self.transforms(x)
            if self.is_mps_available:
                x = x.to(self.device)
        x = self.encoder(x)
        out = []
        for gene in self.genes:
            out.append(getattr(self, gene)(x))
        if len(out) == 1:
            return out[0]
        return torch.cat(out, dim=1)

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.common_step(batch, batch_idx)
        self.log_dict({"training " + self.error_metric_name: loss}, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.common_step(batch, batch_idx)
        self.log_dict({"validation " + self.error_metric_name: loss}, on_step=True, on_epoch=True)
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
        if self.current_loss.abs() < self.best_loss.abs():
            self.best_loss = self.current_loss
            torch.save(self.state_dict(), self.out_path + "/best_model.pth")
            self.best_val_epoch = self.val_epoch_counter
            if self.logging:
                wandb.run.summary["best_val_loss"] = self.best_loss
                wandb.run.summary["best_val_loss_avg"] = self.best_loss / self.num_training_batches
                wandb.run.summary["best_val_epoch"] = self.best_val_epoch

        self.y_hats = torch.cat(self.y_hats, dim=0)
        self.ys = torch.cat(self.ys, dim=0)
        pearsons = self.pearson(self.y_hats, self.ys)
        self.pearson_values.append(pearsons)
        self.val_epoch_counter += 1

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
        best_pearson_epoch_dict = {self.genes[i]: best_pearson_epochs[i] for i in range(len(self.genes))}
        if not self.logging:
            return
        wandb.run.summary["best_pearsons"] = best_pearson_dict
        wandb.run.summary["best_pearson_epochs"] = best_pearson_epoch_dict

    def set_num_training_batches(self, num_batches):
        self.num_training_batches = num_batches

