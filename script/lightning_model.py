from torch.optim import Adam
from loss_functions import SparsityLoss, CompositeLoss
import torch
import lightning as L
import torch.nn as nn
import torchmetrics
import wandb

# lightning module
class LightiningNN(L.LightningModule):
    def __init__(self, genes, encoder, pretrained_out_dim, middel_layer_features, error_metric_name="MSELoss", freeze_pretrained=False):
        super(LightiningNN, self).__init__()

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
        self.mse = torchmetrics.MeanSquaredError()
        self.SparseLoss = SparsityLoss("encoder.layer4.1", self)
        #self.loss = CompositeLoss([self.pearson, self.mse, self.SparseLoss])
        self.loss = self.pearson
        self.optimizer = Adam(self.parameters(), lr=0.01)
        self.error_metric_name = error_metric_name

    def forward(self, x):
        x = self.encoder(x)
        out = []
        for gene in self.genes:
            out.append(getattr(self, gene)(x))
        if len(out) == 1:
            return out[0]
        return torch.cat(out, dim=1)

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        wandb.log({"training " + self.error_metric_name: loss})
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        wandb.log({"validation " + self.error_metric_name: loss})
        return loss

    def common_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        if torch.isnan(loss):
            print("loss is nan")
            exit(1)
        return loss


    def configure_optimizers(self):
        return self.optimizer