import wandb
wandb.login()
from lit_config import config
def init_wandb():
    run = wandb.init(
        # Set the project where this run will be logged
        project=config["project"],
        # Track hyperparameters and run metadata
        config=config,
        dir="../wandb/"
    )

