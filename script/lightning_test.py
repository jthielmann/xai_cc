from lightning_model import LightiningNN
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from config import config
wandb_logger = WandbLogger(project=config["project"])
from wandb_setup import init_wandb
init_wandb()
from lightning_STDataModule import STDataModule
import torchvision.models as models

trainer = L.Trainer(max_epochs=config["epochs"], logger=wandb_logger)
data_module = STDataModule(["RUBCNL"], ["p007", "p014", "p016", "p020", "p025"], ["p009", "p013"], None, data_dir="../data/jonas/Training_Data")
data_module.setup("fit")
model = LightiningNN(["RUBCNL"], models.resnet18(weights=models.ResNet18_Weights.DEFAULT), 1000, 64)
trainer.fit(model=model, datamodule=data_module)

