import sys, traceback

sys.path.insert(0, '..')
import os
import random
import wandb
from torchvision import transforms as T
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from script.data_processing.process_csv import get_dino_csv
from lightning.pytorch.tuner.tuning import Tuner

# Project-specific imports
from script.data_processing.data_loader import get_dino_dataset
from lightly.transforms.dino_transform import DINOTransform
from script.data_processing.stain_normalization import (
    StainNormalizeReinhard,
    ReinhardParams,
    StainThenDINO,
)
from script.model.lit_dino import DINO

from main_utils import ensure_free_disk_space, parse_args, parse_yaml_config, read_config_parameter, get_sweep_parameter_names


def _train(cfg: dict):
    # Initialise W&B run
    run = wandb.init(
        project=cfg.get("project", "dino"),
        config=cfg,
        mode="online" if cfg.get("log_to_wandb", False) else "disabled"
    )

    # Decide normalization: if HF normalization is enabled in the model, skip here to avoid double-normalize
    if cfg.get("use_hf_normalize", False):
        normalize_tfm = None
    else:
        mean = cfg.get("mean", [0.7406, 0.5331, 0.7059])
        std = cfg.get("std", [0.1651, 0.2174, 0.1574])
        normalize_tfm = T.Normalize(mean=mean, std=std)

    # Optional stain normalization before DINO augs
    stain_tfm = None
    if cfg.get("use_stain_norm", False):
        method = str(cfg.get("stain_norm_method", "reinhard")).lower()
        if method == "reinhard":
            tgt_mean = tuple(cfg.get("stain_target_means_lab", [50.0, 0.0, 0.0]))
            tgt_std = tuple(cfg.get("stain_target_stds_lab", [15.0, 5.0, 5.0]))
            stain_tfm = StainNormalizeReinhard(
                ReinhardParams(target_means_lab=tgt_mean, target_stds_lab=tgt_std)
            )
        else:
            raise ValueError(f"Unknown stain_norm_method: {method}")

    # Build reference augmentation policy if requested
    use_ref_aug = bool(cfg.get("use_ref_aug_policy", False))
    aug_params = {
        'global_crop_size': cfg.get("global_crop_size", cfg.get("image_size_global", 224)),
        'global_crop_scale': tuple(cfg.get("global_crop_scale", (0.4, 1.0))),
        'local_crop_size': cfg.get("local_crop_size", cfg.get("image_size_local", 96)),
        'local_crop_scale': tuple(cfg.get("local_crop_scale", (0.05, 0.4))),
        'n_local_views': cfg.get("n_local_views", 6),
        'hf_prob': cfg.get("hf_prob", 0.5),
        'vf_prob': cfg.get("vf_prob", 0.0),
        'rr_prob': cfg.get("rr_prob", 0.0),
        'rr_degrees': cfg.get("rr_degrees", None),
        'cj_prob': cfg.get("cj_prob", 0.8),
        'cj_strength': cfg.get("cj_strength", 0.5),
        'cj_bright': cfg.get("cj_bright", 0.8),
        'cj_contrast': cfg.get("cj_contrast", 0.8),
        'cj_sat': cfg.get("cj_sat", 0.4),
        'cj_hue': cfg.get("cj_hue", 0.2),
        'random_gray_scale': cfg.get("random_gray_scale", 0.2),
        'gaussian_blur': tuple(cfg.get("gaussian_blur", (1.0, 0.1, 0.5))),
        'solarization_prob': cfg.get("solarization_prob", 0.2),
    }
    if use_ref_aug:
        # Typical DINOv3 policy (per-view):
        aug_params.update({
            'global_crop_scale': (0.4, 1.0),
            'local_crop_scale': (0.05, 0.4),
            'n_local_views': 6,
            'hf_prob': 0.5,
            'vf_prob': 0.0,
            'rr_prob': 0.0,
            'rr_degrees': None,
            'cj_prob': 0.8,
            'cj_strength': 0.5,
            'cj_bright': 0.8,
            'cj_contrast': 0.8,
            'cj_sat': 0.4,
            'cj_hue': 0.2,
            'random_gray_scale': 0.2,
            'gaussian_blur': (1.0, 0.1, 0.5),   # g1=1.0, g2=0.1, local=0.5
            'solarization_prob': 0.2,           # applied on 2nd global view by DINOTransform
        })
    # Create DINO transforms: apply augs before normalization
    dino_tfm = DINOTransform(
        global_crop_size=aug_params['global_crop_size'],
        global_crop_scale=aug_params['global_crop_scale'],
        local_crop_size=aug_params['local_crop_size'],
        local_crop_scale=aug_params['local_crop_scale'],
        n_local_views=aug_params['n_local_views'],
        hf_prob=aug_params['hf_prob'],
        vf_prob=aug_params['vf_prob'],
        rr_prob=aug_params['rr_prob'],
        rr_degrees=aug_params['rr_degrees'],
        cj_prob=aug_params['cj_prob'],
        cj_strength=aug_params['cj_strength'],
        cj_bright=aug_params['cj_bright'],
        cj_contrast=aug_params['cj_contrast'],
        cj_sat=aug_params['cj_sat'],
        cj_hue=aug_params['cj_hue'],
        random_gray_scale=aug_params['random_gray_scale'],
        gaussian_blur=aug_params['gaussian_blur'],
        solarization_prob=aug_params['solarization_prob'],
        normalize=normalize_tfm
    )

    # Wrap with stain normalization if requested
    transform = StainThenDINO(stain_tfm, dino_tfm) if stain_tfm is not None else dino_tfm

    # Build datasets
    file_path_train, file_path_val = get_dino_csv(0.8, "../data/NCT-CRC-HE-100K/")
    debug = cfg.get("debug")
    sanity_run = bool(cfg.get("sanity_run", False))
    # Cap dataset size for sanity runs (tiny subset)
    # Users can override with `sanity_max_samples`; defaults keep it very small.
    sanity_max = int(cfg.get("sanity_max_samples", 1024)) if sanity_run else None

    # make sure that we can debug on 16GB ram macbook
    if cfg.get("debug") and torch.backends.mps.is_available():
        cfg["batch_size"] = 32

    # no bins for NCT-CRC-HE-100K for now as it is already somewhat balanced
    train_dataset = get_dino_dataset(
        csv_path=file_path_train,
        dino_transforms=transform,
        max_len=(sanity_max if sanity_run else (cfg.get("batch_size") if debug else None)),
        bins=None,
        device_handling=False,
    )
    val_dataset = get_dino_dataset(
        csv_path=file_path_val,
        dino_transforms=transform,
        max_len=(min(256, sanity_max) if sanity_run and sanity_max is not None else (cfg.get("batch_size") if debug else None)),
        bins=None,
        device_handling=False,
    )

    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.get("batch_size"), shuffle=True, drop_last=True, num_workers=cfg.get("num_workers", 0))
    dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.get("batch_size"), shuffle=False, drop_last=False, num_workers=cfg.get("num_workers", 0))
    # Precision/accelerator
    use_gpu = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    accelerator = "gpu" if (use_gpu or use_mps) else exit("No GPU available")
    precision = None
    if cfg.get("precision_auto", True):
        if use_gpu and torch.cuda.is_bf16_supported():
            precision = "bf16-mixed"
        elif use_gpu:
            precision = "16-mixed"
        else:
            precision = 32
    else:
        precision = cfg.get("precision", 32)

    grad_clip_val = cfg.get("grad_clip_val", 0.0)
    grad_clip_algo = cfg.get("grad_clip_algo", "norm")

    trainer = L.Trainer(
        devices=1,
        accelerator=accelerator,
        precision=precision,
        gradient_clip_val=grad_clip_val,
        gradient_clip_algorithm=grad_clip_algo,
        sync_batchnorm=cfg.get("use_sync_batchnorm", False),
        logger=WandbLogger(run.project)
    )
    model = DINO(cfg)
    # Optional torch.compile (backbone) — safe when CUDA is available
    if cfg.get("use_torch_compile", False):
        try:
            import torch as _torch
            target = cfg.get("compile_target", "backbone")
            mode = cfg.get("compile_mode", None)
            if target == "model":
                model = _torch.compile(model, mode=mode)
            elif target == "backbone":
                model.student_backbone = _torch.compile(model.student_backbone, mode=mode)
            elif target == "head":
                model.student_head = _torch.compile(model.student_head, mode=mode)
        except Exception as e:
            print(f"torch.compile not applied: {e}")
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

    # Optional high-res stage
    if cfg.get("run_high_res_stage", False):
        hi_epochs = int(cfg.get("high_res_epochs", 1 if debug else 5))
        # Rebuild transforms with larger sizes
        hi_norm = None if cfg.get("use_hf_normalize", False) else T.Normalize(mean=cfg.get("mean", [0.7406, 0.5331, 0.7059]), std=cfg.get("std", [0.1651, 0.2174, 0.1574]))
        hi_tfm = DINOTransform(
            global_crop_size=cfg.get("high_global_crop_size", cfg.get("image_size_global", 336)),
            global_crop_scale=tuple(cfg.get("global_crop_scale", (0.4, 1.0))),
            local_crop_size=cfg.get("high_local_crop_size", cfg.get("image_size_local", 128)),
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
            normalize=hi_norm,
        )
        # Wrap stain tfm if requested
        hi_transform = StainThenDINO(stain_tfm, hi_tfm) if cfg.get("use_stain_norm", False) else hi_tfm
        train_dataset_hi = get_dino_dataset(
            csv_path=file_path_train,
            dino_transforms=hi_transform,
            max_len=(sanity_max if sanity_run else (cfg.get("batch_size") if debug else None)),
            bins=None,
            device_handling=False,
        )
        val_dataset_hi = get_dino_dataset(
            csv_path=file_path_val,
            dino_transforms=hi_transform,
            max_len=(min(256, sanity_max) if sanity_run and sanity_max is not None else (cfg.get("batch_size") if debug else None)),
            bins=None,
            device_handling=False,
        )
        dl_train_hi = torch.utils.data.DataLoader(train_dataset_hi, batch_size=cfg.get("batch_size"), shuffle=True, drop_last=True, num_workers=cfg.get("num_workers", 0))
        dl_val_hi = torch.utils.data.DataLoader(val_dataset_hi, batch_size=cfg.get("batch_size"), shuffle=False, drop_last=False, num_workers=cfg.get("num_workers", 0))
        model.set_num_training_batches(len(dl_train_hi))
        trainer.fit_loop.max_epochs += hi_epochs
        trainer.fit(model=model, train_dataloaders=dl_train_hi, val_dataloaders=dl_val_hi)

    wandb_run_id = run.id
    best_ckpt = getattr(getattr(trainer, "checkpoint_callback", None), "best_model_path", "")

    if best_ckpt and os.path.exists(best_ckpt):
        artifact = wandb.Artifact("model-best", type="model")
        artifact.add_file(best_ckpt, name="model.ckpt")
        run.log_artifact(artifact)
        run.summary["best_model_artifact"] = artifact.name

    out_dir = cfg.get("out_path", ".")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "wandb_run_id.txt"), "w") as f:
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
