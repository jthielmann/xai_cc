import torch
print(torch.__version__)
from lit_config import lit_config
from cluster_functions import cluster_explanations_genes_loop
from model import load_model
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
import json
import os

def main():
    wandb_logger = WandbLogger(project=lit_config["project"])

    debug = lit_config["debug"]

    wandb.login()

    # clustering only works on cpu
    device = "cpu"
    model = load_model(lit_config["output_dir"], lit_config["output_dir"] + "/best_model.pth", squelch=True).to(device)

    print("train data clustering start")
    results_paths_train = cluster_explanations_genes_loop(model, lit_config["data_dir"], out_dir, genes=lit_config["genes"],
                                                          debug=debug, samples=lit_config["train_samples"])
    for i in range(len(results_paths_train)):
        wandb.log({"clustering train restults for " + lit_config["genes"][i]: wandb.Image(figure_paths[i])})
    print("valid data clustering start")
    results_paths_valid = cluster_explanations_genes_loop(model, lit_config["data_dir"], out_dir, genes=lit_config["genes"],
                                                          debug=debug, samples=lit_config["val_samples"])
    for i in range(len(results_paths_valid)):
        wandb.log({"clustering valid restults for " + lit_config["genes"][i]: wandb.Image(figure_paths[i])})


def main():
    wandb_logger = WandbLogger(project=lit_config["project"])

    debug = lit_config["debug"]

    wandb.login()
    for dir in os.path
        for epoch_count in lit_config["epochs"]:
            out_dir = "../models/" + "lit_testing"
            encoder, encoder_weights = get_encoder(lit_config["encoder_type"])
            model = LightiningNN(genes=lit_config["genes"], encoder=encoder, pretrained_out_dim=2048,
                                 middel_layer_features=64,
                                 error_metric_name=lit_config["error_metric_name"], out_path=out_dir,
                                 freeze_pretrained=lit_config["freeze_pretrained"], epochs=epoch_count,
                                 learning_rate=learning_rate)
            model_name = model.encoder.__class__.__name__
            name = get_name(epoch_count, model_name, learning_rate, lit_config["encoder_type"], lit_config["error_metric_name"],
                            lit_config["freeze_pretrained"], lit_config["dataset"], lit_config["batch_size"])

            if os.path.exists(out_dir) and not debug:
                print(out_dir, "out_dir already exists")
                continue
            else:
                os.makedirs(out_dir, exist_ok=True)
            with wandb.init(
                # Set the project where this run will be logged
                project=lit_config["project"],
                # Track hyperparameters and run metadata
                config=lit_config,
                dir="../wandb/",
                id=None,
                name=name
            ):
                try:
                    print("loading data")
                    # free the memory because clustering currently loads it aswell but is not tied to lightning
                    data_module = STDataModule(lit_config["genes"], lit_config["train_samples"], lit_config["val_samples"], lit_config["test_samples"], lit_config["data_dir"], lit_config["batch_size"])
                    data_module.setup("fit")
                    print("data loaded")
                    trainer = L.Trainer(max_epochs=epoch_count, logger=wandb_logger, overfit_batches=0.02, log_every_n_steps=1)
                    if model.device == "cuda":
                        torch.set_float32_matmul_precision("medium")
                    model.set_num_training_batches(len(data_module.train_dataloader()))
                    torch.onnx.export(model, torch.randn(1, 3, 224, 224), "./lightning_logs" + "/model.onnx")
                    wandb.save("./lightning_logs" + "/model.onnx")
                    trainer.fit(model=model, datamodule=data_module)
                    data_module.free_memory()

                    results_file_name = out_dir + "/results.csv"
                    if not debug and os.path.exists(results_file_name):
                        os.remove(results_file_name)
                    if not os.path.exists(results_file_name):
                        for patient in lit_config["val_samples"]:
                            generate_results_patient(model, model.device, lit_config["data_dir"], patient, lit_config["genes"], results_file_name)

                    figure_paths = generate_hists(model, out_dir, results_file_name)

                    figures = []
                    for path in figure_paths:
                        figures.append(wandb.Image(path))

                    for i in range(len(lit_config["genes"])):
                        wandb.log({"results for " + lit_config["genes"][i]: wandb.Image(figure_paths[i])})
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

                except Exception as e:
                    print("Exception caught:", e)
                    wandb.finish()
                    exit(1)


if __name__ == "__main__":
    main()
