import sys, traceback

sys.path.insert(0, '..')
import os
import random
import wandb
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from script.data_processing.process_csv import get_dino_csv
from lightning.pytorch.tuner.tuning import Tuner

# Project-specific imports
from script.data_processing.data_loader import get_dino_dataset
from lightly.transforms.dino_transform import DINOTransform
from script.model.lit_model import DINO

from main_utils import ensure_free_disk_space, parse_args, parse_yaml_config, read_config_parameter, get_sweep_parameter_names


def _train(cfg: dict):
    # Initialise W&B run
    run = wandb.init(
        project=cfg.get("project", "dino"),
        config=cfg,
        mode="online" if cfg.get("log_to_wandb", False) else "disabled"
    )

    # Create transforms
    transform = DINOTransform(
        global_crop_size=cfg.get("global_crop_size", 224),
        global_crop_scale=tuple(cfg.get("global_crop_scale", (0.4, 1.0))),
        local_crop_size=cfg.get("local_crop_size", 96),
        local_crop_scale=tuple(cfg.get("local_crop_scale", (0.05, 0.4))),
        n_local_views=cfg.get("n_local_views", 6),
        hf_prob=cfg.get("hf_prob", 0.5),
        vf_prob=cfg.get("vf_prob", 0.0),
        rr_prob=cfg.get("rr_prob", 0.0),
        rr_degrees=cfg.get("rr_degrees", None),
        cj_prob=cfg.get("cj_prob", 0.8),
        cj_strength=cfg.get("cj_strength", 0.5),
        cj_bright=cfg.get("cj_bright", 0.8),
        cj_contrast=cfg.get("cj_contrast", 0.8),
        cj_sat=cfg.get("cj_sat", 0.4),
        cj_hue=cfg.get("cj_hue", 0.2),
        random_gray_scale=cfg.get("random_gray_scale", 0.2),
        gaussian_blur=tuple(cfg.get("gaussian_blur", (1.0, 0.1, 0.5))),
        solarization_prob=cfg.get("solarization_prob", 0.2),
        normalize=cfg.get("normalize", None)
    )

    # Build datasets
    file_path_train, file_path_val = get_dino_csv(0.8, "../data/NCT-CRC-HE-100K/")
    debug = cfg.get("debug")

    # make sure that we can debug on 16GB ram macbook
    if cfg.get("debug") and torch.backends.mps.is_available():
        cfg["batch_size"] = 32

    # no bins for NCT-CRC-HE-100K for now as it is already somewhat balanced
    train_dataset = get_dino_dataset(csv_path=file_path_train, dino_transforms=transform,
                                     max_len=cfg.get("batch_size") if debug else None, bins=None, device_handling=False)
    val_dataset = get_dino_dataset(csv_path=file_path_val, dino_transforms=transform,
                                   max_len=cfg.get("batch_size") if debug else None, bins=None, device_handling=False)

    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.get("batch_size"), shuffle=True, drop_last=True, num_workers=cfg.get("num_workers", 0))
    dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.get("batch_size"), shuffle=False, drop_last=False, num_workers=cfg.get("num_workers", 0))
    accelerator = "gpu" if torch.cuda.is_available() or torch.backends.mps.is_available() else exit("No GPU available")
    trainer = L.Trainer(devices=1, accelerator=accelerator, logger=WandbLogger(run.project))
    model = DINO(cfg)
    model.set_num_training_batches(len(dataloader_train))
    tuner = Tuner(trainer)
    if debug:
        # Use fixed learning rate for debugging to avoid tuning time
        suggested_lr = 1e-4
    else:
        lr_finder = tuner.lr_find(model, train_dataloaders=dataloader_train, num_training=300 if not debug else 30)
        suggested_lr = lr_finder.suggestion()
        print(f"Suggested LR: {lr_finder.suggestion()}")
    model.update_lr(suggested_lr)
    trainer.fit_loop.max_epochs = cfg.get("epochs") if not debug else 2
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    wandb_run_id = run.id
    best_ckpt = trainer.checkpoint_callback.best_model_path

    artifact = wandb.Artifact("model-best", type="model")
    artifact.add_file(best_ckpt, name="model.ckpt")
    run.log_artifact(artifact)
    run.summary["best_model_artifact"] = artifact.name

    with open(cfg["out_path"] + "/" + "wandb_run_id.txt", "w") as f:
        f.write(wandb_run_id)

def _sweep_run():
    run = wandb.init()
    cfg = dict(run.config)
    ensure_free_disk_space(cfg.get("out_path", "."))
    _train(cfg)


def main():
    args = parse_args()
    raw_cfg = parse_yaml_config(args.config)

    params = raw_cfg.get("parameters", {})
    is_sweep = any(isinstance(p, dict) and "values" in p for p in params.values())
    if is_sweep:
        # Build sweep config automatically based on 'values'
        params["sweep_parameter_names"] = {"values": [get_sweep_parameter_names(raw_cfg)]}
        sweep_config = {
            "name": read_config_parameter(raw_cfg, "name") if not raw_cfg.get("debug")
            else "debug_" + random.randbytes(4).hex(),
            "method": read_config_parameter(raw_cfg, "method"),
            "metric": read_config_parameter(raw_cfg, "metric"),
            "parameters": read_config_parameter(raw_cfg, "parameters"),
            "project": read_config_parameter(raw_cfg, "project"),
            "description": " ".join(get_sweep_parameter_names(raw_cfg))
        }
        # Determine project and sweep directory
        project = sweep_config["project"] if not read_config_parameter(raw_cfg, "debug") else "debug_" + random.randbytes(4).hex()
        outpath = sweep_config["parameters"].get("out_path").get("value")
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        ensure_free_disk_space(outpath)
        print(f"Project: {project}")
        sweep_dir = os.path.join("..", "wandb_sweep_ids", project, sweep_config["name"])
        os.makedirs(sweep_dir, exist_ok=True)
        sweep_id_file = os.path.join(sweep_dir, "sweep_id.txt")

        # Load or create sweep ID
        if os.path.exists(sweep_id_file):
            with open(sweep_id_file, "r") as f:
                sweep_id = f.read().strip()
            print(f"Loaded existing sweep ID: {sweep_id}")
        else:
            sweep_id = wandb.sweep(sweep_config, project=project)
            with open(sweep_id_file, "w") as f:
                f.write(sweep_id)
            print(f"Initialized new sweep ID: {sweep_id}")

        # Launch agent for the sweep
        wandb.agent(sweep_id, function=_sweep_run, project=project)


if __name__ == "__main__":
    main()
