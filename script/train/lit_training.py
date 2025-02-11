import torch

from script.model.lit_model import LightiningNN

import numpy
import sys
from lightning.pytorch.loggers import WandbLogger
from script.configs.lit_config import lit_config, get_name, get_encoder
import lightning as L

from script.data_processing.lit_STDataModule import STDataModule
from script.data_processing.process_csv import generate_results_patient

from script.train.train import generate_hists
import os
import json
import wandb
import copy
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.profilers import PyTorchProfiler
debug = lit_config["debug"]

def train_model(config=None):
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
        print("saving model to", out_dir)
        model = LightiningNN(genes=lit_config["genes"], encoder=encoder, pretrained_out_dim=2048,
                             middel_layer_features=64,
                             error_metric_name=lit_config["error_metric_name"], out_path=out_dir,
                             freeze_pretrained=lit_config["freeze_pretrained"], epochs=epochs,
                             learning_rate=learning_rate, use_transforms=lit_config["use_transforms_in_model"], logging=not debug)

        if os.path.exists(out_dir) and not debug:
            raise ValueError(out_dir, "out_dir already exists")
        else:
            os.makedirs(out_dir, exist_ok=True)

        print("loading data")
        # free the memory because clustering currently loads it aswell but is not tied to lightning
        data_module = STDataModule(lit_config["genes"], lit_config["train_samples"], lit_config["val_samples"],
                                   lit_config["test_samples"], lit_config["data_dir"], lit_config["num_workers"], lit_config["use_transforms_in_model"],
                                   lit_config["batch_size"], lit_config["debug"])
        data_module.setup("fit")
        print("length of train_dataset", len(data_module.train_dataset))
        print("data loaded")
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]) as p:
            profiler = PyTorchProfiler(record_module_names=True, export_to_chrome=True, profile_memory=True, filename=out_dir + "/profiler_results.json")
            trainer = L.Trainer(max_epochs=epochs, logger=wandb_logger, log_every_n_steps=1,
                                enable_checkpointing=False, callbacks=[EarlyStopping(monitor="validation " + lit_config["error_metric_name"], mode="min", patience=10)],
                                profiler=profiler)

            if model.device == "cuda":
                torch.set_float32_matmul_precision("medium")

            train_loader = data_module.train_dataloader()

            val_loader = data_module.val_dataloader()
            model.set_num_training_batches(len(train_loader))
            #torch.onnx.dynamo_export(model, torch.randn(1, 3, lit_config["image_size"], lit_config["image_size"]))
            # this does not like the v2.resize apparently
            if not lit_config["use_transforms_in_model"]:
                torch.onnx.export(model, torch.randn(1, 3, lit_config["image_size"], lit_config["image_size"]), "./lightning_logs" + "/model.onnx")
                wandb.save("./lightning_logs" + "/model.onnx")
            trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            print("saving trace to", out_dir + "/chrome_trace.json")
            #profiler.export_chrome_trace(out_dir + "/chrome_trace.json")
            data_module.free_memory()
            if debug:
                wandb.finish()
                exit(0)
            results_file_name = out_dir + "/results.csv"
            if not debug and os.path.exists(results_file_name):
                os.remove(results_file_name)
            if not os.path.exists(results_file_name):
                for patient in lit_config["val_samples"]:
                    generate_results_patient(model, model.device, lit_config["data_dir"], patient, lit_config["genes"],
                                             results_file_name)

            figure_paths = generate_hists(model, out_dir, results_file_name)

            figures = []
            for path in figure_paths:
                figures.append(wandb.Image(path))

            for i in range(len(lit_config["genes"])):
                wandb.log({"results for " + lit_config["genes"][i]: wandb.Image(figure_paths[i])})
            wandb.log({"val_samples": lit_config["val_samples"]})
            wandb.log({"train_samples": lit_config["train_samples"]})
            wandb.finish()
            str_config = copy.deepcopy(lit_config)
            for key in str_config.keys():
                if isinstance(str_config[key], list):
                    new_key = ""
                    for item in str_config[key]:
                        new_key += str(item) + "_"
                    str_config[key] = new_key
                else:
                    str_config[key] = str(str_config[key])
            out_file = open(out_dir + "/config.json", "w")
            json.dump(str_config, out_file)
            open(out_dir + "/clustering_job", "w").close()


if __name__ == "__main__":
    print("python version:", sys.version)
    print("numpy version:", numpy.version.version)
    print("torch version:", torch.__version__)
