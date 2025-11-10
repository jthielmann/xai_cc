import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from script.main_utils import ensure_free_disk_space
from script.configs.dataset_config import get_dataset_cfg
from script.data_processing.data_loader import get_dataset
from script.model.model_factory import get_encoder
from script.data_processing.image_transforms import get_eval_transforms


# Always use float16 for exported embeddings
EMBEDDINGS_DTYPE_STR = "float16"
EMBEDDINGS_NP_DTYPE = np.float16


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    raise RuntimeError("accelerator required: cuda or mps")  # why: enforce fast export on accelerators only


def prepare_embeddings_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(cfg)
    merged.update(get_dataset_cfg(merged))

    emb_path = "../embeddings"
    merged["embeddings_path"] = emb_path

    os.makedirs(emb_path, exist_ok=True)
    ensure_free_disk_space(emb_path)

    dump_cfg = {k: v for k, v in merged.items() if not str(k).startswith("_")}
    with open(os.path.join(emb_path, "config"), "w") as handle:
        yaml.safe_dump(
            dump_cfg,
            handle,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
    return merged


def build_encoder(encoder_type: str, device: torch.device) -> torch.nn.Module:
    if not encoder_type:
        raise ValueError("Config must set 'encoder_type' for embeddings export.")
    encoder = get_encoder(encoder_type)
    encoder.to(device)
    return encoder


def convert_model_output_to_tokens(
    out: torch.Tensor, drop_cls_token: bool
) -> Tuple[torch.Tensor, tuple]:
    # Normalize common container types
    if isinstance(out, (list, tuple)) and len(out) > 0:
        out = out[0]
    if hasattr(out, "last_hidden_state"):
        out = out.last_hidden_state  # (B, T, D)

    grid_hw = None
    if out.ndim == 4:
        # (B, C, H, W) → (B, T, D) with T=H*W, D=C
        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1).reshape(B, H * W, C)
        grid_hw = (int(H), int(W))
    elif out.ndim == 3 and drop_cls_token and out.size(1) > 1:
        out = out[:, 1:, :]
    elif out.ndim == 2:
        out = out[:, None, :]
    elif out.ndim not in (2, 3):
        out = torch.flatten(out, 2)
    return out, grid_hw


def update_train_stats(stats_accum: Dict[str, Any], arr_f16: np.ndarray) -> None:
    # Compute numerically stable stats in float64 but from float16 source
    flat64 = arr_f16.reshape(-1, arr_f16.shape[-1]).astype(np.float64, copy=False)
    if stats_accum.get("sum") is None:
        D = flat64.shape[1]
        stats_accum["sum"] = np.zeros((D,), dtype=np.float64)
        stats_accum["sumsq"] = np.zeros((D,), dtype=np.float64)
        stats_accum["count"] = 0
    stats_accum["sum"] += flat64.sum(axis=0)
    stats_accum["sumsq"] += (flat64 * flat64).sum(axis=0)
    stats_accum["count"] += flat64.shape[0]


def export_split_embeddings(
    *,
    split: str,
    cfg: Dict[str, Any],
    encoder: torch.nn.Module,
    device: torch.device,
    drop_cls_token: bool,
    epochs: int,
    image_size: int,
    idx_map: Dict[str, Dict[str, Any]],
    stats_accum: Dict[str, Any] = None,
    patient_split_guard: Dict[str, str],
) -> None:
    out_dir = os.path.join(cfg["embeddings_path"], split)
    os.makedirs(out_dir, exist_ok=True)

    bs = int(cfg.get("batch_size"))
    nw = int(cfg.get("num_workers", 2))

    # enforce current shard policy
    granularity = str(cfg.get("embeddings_save_granularity", "batch"))
    if granularity != "batch":
        raise NotImplementedError("embeddings_save_granularity != 'batch' not supported")  # why: keep IO/mem predictable

    for epoch in range(int(epochs)):
        tfm = get_eval_transforms(image_size=image_size)

        # Build dataset yielding (img, y, patient, tile_path)
        # require explicit metadata sources from config
        gene_csv = cfg["gene_data_filename"]
        ds = get_dataset(
            data_dir=cfg["data_dir"],
            genes=None,
            transforms=tfm,
            samples=cfg.get(f"{split}_samples"),
            only_inputs=False,
            gene_data_filename=gene_csv,
            meta_data_dir=cfg.get("meta_data_dir"),
            return_floats=False,
            return_edges=False,
            return_patient_and_tilepath=True,
        )

        loader = DataLoader(
            ds,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=(device.type == "cuda"),
        )

        encoder.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(loader):
                # Expected shape: (img, y, patient, tile_path)
                if isinstance(batch, (list, tuple)) and len(batch) >= 4:
                    imgs, _y, patients, tiles = batch
                else:
                    raise RuntimeError(
                        "Dataset must return (img, y, patient, tile_path) for embeddings export."
                    )

                imgs = imgs.to(device, non_blocking=True)
                out = encoder(imgs)
                t, grid_hw = convert_model_output_to_tokens(out, drop_cls_token)

                # Always export embeddings as float16
                arr_f16 = (
                    t.detach().cpu().to(dtype=torch.float32).numpy().astype(EMBEDDINGS_NP_DTYPE, copy=False)
                )

                tiles_arr = np.array(list(tiles))
                pats_arr = np.array(list(patients))

                fn = f"emb_ep{epoch:04d}_b{b_idx:06d}_tok.npz"
                out_path = os.path.join(out_dir, fn)
                save_kwargs: Dict[str, Any] = {
                    "embeddings": arr_f16,
                    "tiles": tiles_arr,
                    "patients": pats_arr,
                    "split": np.array(split),
                    "epoch": np.array(epoch, dtype=np.int32),
                    "dtype": np.array(EMBEDDINGS_DTYPE_STR),
                    "drop_cls_token": np.array(bool(drop_cls_token)),
                }

                # Record token spatial info when applicable
                if grid_hw is not None:
                    save_kwargs["grid_hw"] = np.array(grid_hw, dtype=np.int32)
                    save_kwargs["token_idx"] = np.arange(grid_hw[0] * grid_hw[1], dtype=np.int32)
                else:
                    save_kwargs["token_idx"] = np.arange(arr_f16.shape[1], dtype=np.int32)

                np.savez_compressed(out_path, **save_kwargs)

                # Update leakage guard: ensure patients unique across splits
                for p in list(patients):
                    prev = patient_split_guard.get(p)
                    if prev is None:
                        patient_split_guard[p] = split
                    elif prev != split:
                        raise RuntimeError(
                            f"Patient {p!r} appears in multiple splits: {prev} and {split}"
                        )

                # Update index for random access
                for row_idx, tile in enumerate(list(tiles)):
                    entry = {
                        "file": os.path.relpath(out_path, cfg["embeddings_path"]),
                        "offset": int(row_idx),
                        "split": split,
                        "epoch": int(epoch),
                        "patient": str(pats_arr[row_idx]),
                    }
                    entry["tokens"] = int(arr_f16.shape[1])
                    idx_map[str(tile)] = entry

                # Update stats from train split only
                if stats_accum is not None and split == "train":
                    update_train_stats(stats_accum, arr_f16)

                print(
                    f"[embeddings] wrote split={split} epoch={epoch} batch={b_idx} → {out_path}"
                )
