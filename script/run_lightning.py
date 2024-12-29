import lightning as L
from lightning.pytorch.loggers import WandbLogger
from config import config

wandb_logger = WandbLogger(project=config["project"])
from wandb_setup import init_wandb
init_wandb()
from lightning_STDataModule import STDataModule
from process_csv import generate_results

from lightning.pytorch.callbacks import Callback
from generate_plots import generate_hists
import wandb

class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
        # calculate pearson correlation


trainer = L.Trainer(max_epochs=config["epochs"], logger=wandb_logger, overfit_batches=0.02, log_every_n_steps=1, callbacks=[MyPrintingCallback()])
data_module = STDataModule(config["genes"], config["train_samples"], config["val_samples"], config["test_samples"], config["data_dir"], config["batch_size"])
data_module.setup("fit")
model = config["model"]
trainer.fit(model=model, datamodule=data_module)

results_file_name = config["output_dir"] + "/results.csv"
for patient in config["val_samples"]:
    generate_results(model, model.device, config["data_dir"], patient, config["genes"], results_file_name)

figure_paths = generate_hists(config["output_dir"], config["output_dir"] + "/best_model.pth", model.device, results_file_name)

figures = []
for path in figure_paths:
    figures.append(wandb.Image(path))


for i in range(len(config["genes"])):
    wandb.log({"restults for " + config["genes"][i]: wandb.Image(figure_paths[i])})
wandb.finish()


# clustering
