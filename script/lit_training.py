import torch
from lightning.pytorch.loggers import WandbLogger
from lit_config import config
if config["model"].device == "cuda":
    torch.set_float32_matmul_precision("medium")
import lightning as L

wandb_logger = WandbLogger(project=config["project"])
from wandb_setup import init_wandb
init_wandb()
from lightning_STDataModule import STDataModule
from process_csv import generate_results

from generate_plots import generate_hists
import wandb

debug = True

try:
    trainer = L.Trainer(max_epochs=config["epochs"], logger=wandb_logger, overfit_batches=0.02, log_every_n_steps=1)
    data_module = STDataModule(config["genes"], config["train_samples"], config["val_samples"], config["test_samples"], config["data_dir"], config["batch_size"])
    data_module.setup("fit")
    if debug:
        data_module.train_dataset.dataframe = data_module.train_dataset.dataframe.drop(list(range(10, len(data_module.train_dataset.dataframe))))
    model = config["model"]
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), "./lightning_logs" + "/model.onnx")
    wandb.save("./lightning_logs" + "/model.onnx")
    trainer.fit(model=model, datamodule=data_module)

    results_file_name = config["output_dir"] + "/results.csv"
    for patient in config["val_samples"]:
        generate_results(model, model.device, config["data_dir"], patient, config["genes"], results_file_name)

    figure_paths = generate_hists(model, config["output_dir"], results_file_name)

    figures = []
    for path in figure_paths:
        figures.append(wandb.Image(path))


    for i in range(len(config["genes"])):
        wandb.log({"restults for " + config["genes"][i]: wandb.Image(figure_paths[i])})


    wandb.finish()
except Exception as e:
    print(e)
    wandb.finish()
    exit(1)