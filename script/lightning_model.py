from torch.optim import Adam
from loss_functions import SparsityLoss, CompositeLoss
import torch
import lightning as L
import torch.nn as nn
import torchmetrics
import wandb

# lightning module
class LightiningNN(L.LightningModule):
    def __init__(self, genes, encoder, pretrained_out_dim, middel_layer_features, out_path, error_metric_name, freeze_pretrained):
        super(LightiningNN, self).__init__()
        self.save_hyperparameters()
        # model setup
        self.encoder = encoder
        self.freeze_pretrained = freeze_pretrained
        if self.freeze_pretrained:
            for param in self.encoder.parameters():
                param.requires_grad = False

        for gene in genes:
            setattr(self, gene, nn.Sequential(nn.Linear(pretrained_out_dim, middel_layer_features), nn.ReLU(),
                                              nn.Linear(middel_layer_features, 1)))

        self.genes = genes
        # metrics
        self.pearson = torchmetrics.PearsonCorrCoef(num_outputs=len(genes))
        self.loss = torch.nn.MSELoss()
        self.optimizer = Adam(self.parameters(), lr=0.01)
        self.error_metric_name = error_metric_name
        self.current_loss = torch.Tensor([0])
        self.best_loss = torch.Tensor([float("Inf")])
        self.out_path = out_path
        self.pearson_values = []
        self.y_hats = []
        self.ys = []
        self.val_epoch_counter = 0
        self.best_val_epoch = 0

    def forward(self, x):
        x = self.encoder(x)
        out = []
        for gene in self.genes:
            out.append(getattr(self, gene)(x))
        if len(out) == 1:
            return out[0]
        return torch.cat(out, dim=1)

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.common_step(batch, batch_idx)
        wandb.log({"training " + self.error_metric_name: loss})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.common_step(batch, batch_idx)
        wandb.log({"validation " + self.error_metric_name: loss})
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
        return self.optimizer

    def on_validation_start(self) -> None:
        # in the init function self.device is cpu, so we can do it e.g. here
        self.best_loss = self.best_loss.to(self.device)

    def on_validation_epoch_start(self):
        self.current_loss = torch.Tensor([0]).to(self.device)
        self.y_hats = []
        self.ys = []

    def on_validation_epoch_end(self):
        if self.current_loss.abs() < self.best_loss.abs():
            self.best_loss = self.current_loss
            torch.save(self.state_dict(), self.out_path + "/best_model.pth")
            self.best_val_epoch = self.val_epoch_counter
            wandb.run.summary["best_val_loss"] = self.best_loss
            wandb.run.summary["best_val_epoch"] = self.best_val_epoch

        self.y_hats = torch.cat(self.y_hats, dim=0)
        self.ys = torch.cat(self.ys, dim=0)
        pearsons = self.pearson(self.y_hats, self.ys)
        self.pearson_values.append(pearsons)
        self.val_epoch_counter += 1

    def on_train_epoch_end(self):
        torch.save(self.state_dict(), self.out_path + "/latest.pth")

    def on_train_end(self):
        best_pearsons = []
        best_pearson_epochs = []
        for i in range(len(self.genes)):
            best_epoch = torch.argmax(torch.stack(self.pearson_values)[:,i].abs(), dim=0).item()
            best_pearson_epochs.append(best_epoch)
            best_pearsons.append(round(self.pearson_values[best_epoch][i].item(), 3))
        best_pearson_dict = {self.genes[i]: best_pearsons[i] for i in range(len(self.genes))}
        wandb.run.summary["best_pearsons"] = best_pearson_dict
        best_pearson_epoch_dict = {self.genes[i]: best_pearson_epochs[i] for i in range(len(self.genes))}
        wandb.run.summary["best_pearson_epochs"] = best_pearson_epoch_dict

        wandb.log({"final pearson": self.pearson_values[self.best_val_epoch]})
