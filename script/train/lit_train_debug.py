import matplotlib.pyplot as plt
import torch

from script.model.lit_model import LightiningNN

import numpy
import sys
from script.configs.lit_config import get_name, get_encoder, lit_config
import lightning as L

from script.data_processing.lit_STDataModule import STDataModule
from script.data_processing.process_csv import generate_results_patient

from script.train.generate_hists import generate_hists
import os
import json
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.profilers import PyTorchProfiler
import wandb
from lightning.pytorch.loggers import WandbLogger
from script.data_processing.image_transforms import get_transforms

def train_model(genes, epochs, learning_rate, encoder, encoder_type, error_metric_name, freeze_pretrained, out_dir,
                train_samples, val_samples, data_dir, num_workers, batch_size, debug,
                use_transforms_in_model=True,
                do_hist_generation=False, do_results_generation=False, logger:any=False):
    print("train_samples", train_samples)
    print("val_samples", val_samples)
    print("saving model to", out_dir)

    model = LightiningNN(genes=genes, encoder=encoder, pretrained_out_dim=lit_config["pretrained_out_dim"],
                         middel_layer_features=64,
                         error_metric_name=error_metric_name, out_path=out_dir,
                         freeze_pretrained=freeze_pretrained, epochs=epochs,
                         learning_rate=learning_rate, use_transforms=use_transforms_in_model, logging=not debug)

    os.makedirs(out_dir, exist_ok=True)

    print("loading data")
    # free the memory because clustering currently loads it aswell but is not tied to lightning
    data_module = STDataModule(genes, train_samples, val_samples, None, data_dir, num_workers, use_transforms_in_model, batch_size, debug=debug)
    data_module.setup("fit")
    print("length of train_dataset", len(data_module.train_dataset))
    print("data loaded")
    if debug:
        profiler = PyTorchProfiler(record_module_names=True, export_to_chrome=True, profile_memory=True,
                                on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"))
    else:
        profiler = None
    """profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"),
        record_shapes=True,
        with_stack=True,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    )"""

    trainer = L.Trainer(max_epochs=epochs, logger=logger, log_every_n_steps=1,
                        enable_checkpointing=False, callbacks=[EarlyStopping(monitor="validation " + error_metric_name, mode="min", patience=10)],
                        profiler=profiler, accelerator="gpu")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    model.set_num_training_batches(len(train_loader))
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    data_module.free_memory()
    if do_results_generation:
        results_file_name = out_dir + "/results.csv"
        if os.path.exists(results_file_name):
            os.remove(results_file_name)
        if not os.path.exists(results_file_name):
            if lit_config["use_transforms_in_model"]:
                transforms = None
                max_len = None
            else:
                transforms = get_transforms()
                max_len = 100
            for patient in val_samples:
                generate_results_patient(model, model.device, data_dir, patient, genes, results_file_name, transforms, max_len=max_len)
        if do_hist_generation:
            figure_paths = generate_hists(model, out_dir, results_file_name)

            for path in figure_paths:
                plt.imshow(plt.imread(path))
                plt.show()


        str_config = {}
        str_config["genes"] = genes
        str_config["epochs"] = epochs
        str_config["batch_size"] = batch_size
        str_config["data_dir"] = data_dir
        str_config["freeze_pretrained"] = freeze_pretrained
        str_config["train_samples"] = train_samples
        str_config["val_samples"] = val_samples
        str_config["test_samples"] = None
        str_config["error_metric_name"] = error_metric_name
        str_config["encoder_type"] = encoder_type
        str_config["learning_rate"] = learning_rate
        str_config["use_transforms_in_model"] = use_transforms_in_model
        str_config["num_workers"] = num_workers

        out_file = open(out_dir + "/config.json", "w")
        json.dump(str_config, out_file)
        open(out_dir + "/clustering_job", "w").close()
    print("finished training")


def train_model_sweep(config=None):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        print("no cuda or mps device available")
        exit(1)
    print("train_samples", lit_config["train_samples"])
    print("val_samples", lit_config["val_samples"])
    wandb_logger = WandbLogger(project=lit_config["project"])
    debug = lit_config["debug"]

    with wandb.init(
            # Set the project where this run will be logged
            project=lit_config["project"],
            # Track hyperparameters and run metadata
            config=config,
            dir="../wandb/"
    ) as run:

        config = run.config
        epochs = config["epochs"]
        learning_rate = config["learning_rate"]

        encoder = get_encoder(lit_config["encoder_type"])
        model_name = encoder.__class__.__name__
        name = None if debug else get_name(epochs, model_name, learning_rate, lit_config["encoder_type"],
                                           lit_config["error_metric_name"],
                                           lit_config["freeze_pretrained"], lit_config["dataset"], lit_config["batch_size"])
        run.name = str(epochs) + "_" + str(learning_rate)
        out_dir = "../models/" + run.project + "/"
        out_dir += "lit_testing" if debug else name

        train_model(genes=lit_config["genes"], epochs=epochs,
                    learning_rate=learning_rate,
                    encoder=get_encoder(lit_config["encoder_type"]), encoder_type=lit_config["encoder_type"],
                    error_metric_name=lit_config["error_metric_name"],
                    freeze_pretrained=lit_config["freeze_pretrained"],
                    out_dir=out_dir, train_samples=lit_config["train_samples"],
                    val_samples=lit_config["val_samples"], data_dir=lit_config["data_dir"],
                    num_workers=lit_config["num_workers"], batch_size=lit_config["batch_size"], debug=lit_config["debug"],
                    use_transforms_in_model=lit_config["use_transforms_in_model"],
                    do_hist_generation=False, do_results_generation=False, logger=wandb_logger)


if __name__ == "__main__":
    print("python version:", sys.version)
    print("numpy version:", numpy.version.version)
    print("torch version:", torch.__version__)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("device", device)
    from script.configs.lit_config import lit_config
    train_model(genes=lit_config["genes"], epochs=lit_config["epochs"][0], learning_rate=lit_config["learning_rates"][0],
                encoder=get_encoder(lit_config["encoder_type"]), encoder_type=lit_config["encoder_type"],
                error_metric_name=lit_config["error_metric_name"], freeze_pretrained=lit_config["freeze_pretrained"],
                out_dir="../models/debug", train_samples=lit_config["train_samples"],
                val_samples=lit_config["val_samples"], data_dir=lit_config["data_dir"],
                num_workers=lit_config["num_workers"], batch_size=lit_config["batch_size"], debug=True,
                use_transforms_in_model=lit_config["use_transforms_in_model"],
                do_hist_generation=True, do_results_generation=True)
