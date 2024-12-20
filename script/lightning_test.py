from lightning_model import LightiningNN
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from config import config
wandb_logger = WandbLogger(project=config["project"])
from wandb_setup import init_wandb
init_wandb()
from lightning_STDataModule import STDataModule

trainer = L.Trainer(max_epochs=config["epochs"], logger=wandb_logger, overfit_batches=0.01)
data_module = STDataModule(config["genes"], config["train_samples"], config["val_samples"], config["test_samples"], config["data_dir"])
data_module.setup("fit")
model = config["model"]
trainer.fit(model=model, datamodule=data_module)
