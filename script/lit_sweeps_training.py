import torch
import numpy
import sys
import wandb
from script.train.lit_train_debug import train_model_sweep
from script.configs.lit_sweeps_config import sweep_config
import os
from script.configs.lit_config import lit_config

if __name__ == '__main__':

    print("python version:", sys.version)
    print("numpy version:", numpy.version.version)
    print("torch version:", torch.__version__)
    if lit_config["debug"]:
        project_name = ("debug77717877")
    else:
        project_name = "transforms_lr_epochs_CRC-N19_no_enteric_glial_cells_random_temp_norm"
    sweep_id_path = "../wandb_sweep_ids/" + project_name + "/sweep_id.txt"
    if os.path.exists(sweep_id_path):
        with open(sweep_id_path, "r") as f:
            sweep_id = f.read()
        print("sweep_id:", sweep_id)
    else:
        sweep_id=wandb.sweep(sweep_config, project=project_name)
        os.makedirs("../wandb_sweep_ids/" + project_name, exist_ok=True)
        with open(sweep_id_path, "w") as f:
            f.write(sweep_id)
        print("sweep_id:", sweep_id)
    wandb.agent(sweep_id=sweep_id, function=train_model_sweep, project=project_name)