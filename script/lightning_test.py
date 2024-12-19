import torch.nn as nn
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
import lightning as L
import wandb
wandb.login()

from torch.optim import Adam
from loss_functions import SparsityLoss, CompositeLoss
import torch

# lightning module
class LightiningNN(L.LightningModule):
    def __init__(self, genes, encoder, pretrained_out_dim, middel_layer_features):
        super(LightiningNN, self).__init__()

        # model setup
        self.encoder = encoder
        for gene in genes:
            setattr(self, gene, nn.Sequential(nn.Linear(pretrained_out_dim, middel_layer_features), nn.ReLU(),
                                              nn.Linear(middel_layer_features, 1)))

        self.genes = genes
        # metrics
        self.pearson = torchmetrics.PearsonCorrCoef(num_outputs=len(genes))
        self.mse = torchmetrics.MeanSquaredError()
        self.SparseLoss = SparsityLoss("encoder.layer4.1", self)
        self.loss = CompositeLoss([self.pearson, self.mse, self.SparseLoss])
        self.optimizer = Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        x = self.encoder(x)
        out = []
        for gene in self.genes:
            out.append(getattr(self, gene)(x))
        if len(out) == 1:
            return out[0]
        return torch.cat(out, dim=1)

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx)

    def common_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss


    def configure_optimizers(self):
        return self.optimizer



# trainer
print("logger")
logger = WandbLogger()
print("trainer")
trainer = L.Trainer(max_epochs=1, logger=logger)

from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from data_loader import get_dataset

# data module
class STDataModule(L.LightningDataModule):
    def __init__(self, genes, train_samples, val_samples, test_samples, data_dir="../data/jonas/Training_Data/"):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # mean and std of the whole dataset
            transforms.Normalize([0.7406, 0.5331, 0.7059], [0.1651, 0.2174, 0.1574])
            ])
        self.genes = genes
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = get_dataset(self.data_dir, genes=self.genes, samples=self.train_samples, transforms=self.transforms)
            self.val_dataset = get_dataset(self.data_dir, genes=self.genes, samples=self.val_samples, transforms=self.transforms)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = get_dataset(self.data_dir, genes=self.genes, samples=self.test_samples, transforms=self.transforms)


    def train_dataloader(self, batch_size=64):
        return DataLoader(self.train_dataset, batch_size=batch_size)

    def val_dataloader(self, batch_size=64):
        return DataLoader(self.val_dataset, batch_size=batch_size)

    def test_dataloader(self, batch_size=64):
        return DataLoader(self.test_dataset, batch_size=batch_size)

print("data_module")
data_module = STDataModule(["RUBCNL"], ["p007", "p014", "p016", "p020", "p025"], ["p009", "p013"], None, data_dir="../data/jonas/Training_Data")
data_module.setup("fit")
from torchvision.models import resnet18
model = LightiningNN(["RUBCNL"], resnet18(pretrained=True), 1000, 64)
trainer.fit(model=model, datamodule=data_module)

