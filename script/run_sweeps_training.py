if __name__ == '__main__':

    import torch
    if torch.backends.mps.is_available():
        torch.set_default_dtype(torch.float32)

    import numpy
    import sys

    sys.path.insert(0, '..')
    print("python version:", sys.version)
    print("numpy version:", numpy.version.version)
    print("torch version:", torch.__version__)
    import wandb
    from train.lit_train import train_model_sweep
    from script.configs.lit_config import sweep_config
    import os
    from script.configs.lit_config import lit_config
    import random

    if lit_config["debug"]:
        project_name = "debug_" + random.randbytes(4).hex()
    else:
        project_name = "debug9"
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
