if __name__ == '__main__':

    import torch
    if torch.backends.mps.is_available():
        torch.set_default_dtype(torch.float32)

    import numpy
    import sys

    sys.path.insert(0, '..')
    num_args = len(sys.argv)

    print("python version:", sys.version)
    print("numpy version:", numpy.version.version)
    print("torch version:", torch.__version__)
    import wandb
    from script.dino.dino import train_model_sweep_dino
    from script.configs.dino_config import sweep_config, dino_config
    import os
    import random

    if dino_config["debug"]:
        project_name = "debug_" + random.randbytes(4).hex()
    else:
        project_name = "dino_init"
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
    wandb.agent(sweep_id=sweep_id, function=train_model_sweep_dino, project=project_name)
