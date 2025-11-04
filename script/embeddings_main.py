import os
import sys
from typing import Any, Dict, Tuple, List
import json
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import wandb

# keep repo-local imports stable
sys.path.insert(0, "..")  # why: avoid brittle PYTHONPATH

# avoid OpenMP/TBB clashes and oversubscription
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")  # why: numba vs MKL/Torch
os.environ.setdefault("OMP_NUM_THREADS", "1")  # 1 limits oversubscription
os.environ.setdefault("MKL_NUM_THREADS", "1")  # 1 limits oversubscription
os.environ.setdefault("MPLBACKEND", "Agg")  # why: headless envs

from script.main_utils import parse_args, parse_yaml_config, read_config_parameter, setup_dump_env
from script.embeddings_main_helper import (
    EMBEDDINGS_DTYPE_STR,
    prepare_embeddings_config,
    resolve_device,
    build_encoder,
    export_split_embeddings,
)


def _require(rcfg: Dict[str, Any], key: str):
    v = read_config_parameter(rcfg, key)
    if v is None:
        raise ValueError(f"missing required config '{key}'")
    return v


def _build_runtime_cfg(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for k in ("dataset", "meta_data_dir", "gene_data_filename", "encoder_type", "image_size"):
        cfg[k] = _require(raw_cfg, k)

    debug_val = read_config_parameter(raw_cfg, "debug")
    if debug_val is not None:
        cfg["debug"] = debug_val

    bs_val = read_config_parameter(raw_cfg, "batch_size")
    cfg["batch_size"] = bs_val if bs_val is not None else 32  # 32 balances throughput/memory

    nw_val = read_config_parameter(raw_cfg, "num_workers")
    cfg["num_workers"] = nw_val if nw_val is not None else 2  # 2 is stable on shared FS

    dct = read_config_parameter(raw_cfg, "drop_cls_token")
    cfg["drop_cls_token"] = bool(dct) if dct is not None else True  # prefer spatial tokens

    ep = read_config_parameter(raw_cfg, "embedding_epochs")
    if ep is None:
        ep = read_config_parameter(raw_cfg, "epochs")
    if ep is None:
        raise ValueError("missing required config 'embedding_epochs' or 'epochs'")
    cfg["embedding_epochs"] = ep

    cfg["embeddings_save_granularity"] = read_config_parameter(raw_cfg, "embeddings_save_granularity") or "batch"  # 'batch' avoids many files

    cfg["log_to_wandb"] = bool(read_config_parameter(raw_cfg, "log_to_wandb"))
    if cfg["log_to_wandb"]:
        def _wb(key: str) -> Any:
            v = read_config_parameter(raw_cfg, key)
            if v is None:
                v = read_config_parameter(raw_cfg, f"wandb_{key}")
            if v is None:
                raise ValueError(f"missing wandb '{key}'")
            return v

        cfg["project"] = _wb("project")
        cfg["group"] = _wb("group")
        cfg["run_name"] = _wb("run_name")
        tags_val = read_config_parameter(raw_cfg, "tags") or read_config_parameter(raw_cfg, "wandb_tags")
        if tags_val is not None:
            cfg["tags"] = tags_val

        cfg["wandb_umap_sample_per_split"] = int(read_config_parameter(raw_cfg, "wandb_umap_sample_per_split") or 10000)  # 10000 balances clarity/perf
        cfg["wandb_pca_components"] = int(read_config_parameter(raw_cfg, "wandb_pca_components") or 50)  # 50 denoises + speeds UMAP
        cfg["wandb_umap_neighbors"] = int(read_config_parameter(raw_cfg, "wandb_umap_neighbors") or 30)  # 30 balances local/global
        cfg["wandb_umap_min_dist"] = float(read_config_parameter(raw_cfg, "wandb_umap_min_dist") or 0.1)  # 0.1 keeps clusters compact
        cfg["wandb_umap_seed"] = int(read_config_parameter(raw_cfg, "wandb_umap_seed") or 42)  # 42 reproducibility
        cfg["wandb_plot_size"] = int(read_config_parameter(raw_cfg, "wandb_plot_size") or 224)  # 224 matches encoder input
    return cfg


def _save_index(base: str, idx_map: Dict[str, Dict[str, Any]]) -> None:
    with open(os.path.join(base, "index.json"), "w") as f:
        json.dump(idx_map, f)


def _save_stats(base: str, stats_accum: Dict[str, Any], *, drop_cls_token: bool, image_size: int) -> None:
    if stats_accum.get("count", 0) <= 0:
        return
    total_count = int(stats_accum["count"])
    sum_vec = stats_accum["sum"]
    sumsq_vec = stats_accum["sumsq"]
    mean_vec = (sum_vec / total_count).astype(np.float32)
    var_vec = (sumsq_vec / total_count) - (mean_vec.astype(np.float64) ** 2)
    var_vec = np.maximum(var_vec, 0.0)  # 0.0 avoids negative var from fp errors
    std_vec = np.sqrt(var_vec).astype(np.float32)
    payload = {
        "dtype": EMBEDDINGS_DTYPE_STR,
        "drop_cls_token": drop_cls_token,
        "image_size": image_size,
        "feature_dim": int(mean_vec.shape[0]),
        "count": total_count,
        "mean": mean_vec.tolist(),
        "std": std_vec.tolist(),
    }
    with open(os.path.join(base, "embed_stats.json"), "w") as f:
        json.dump(payload, f)


def _list_npz_files(dir_path: str) -> List[str]:
    return sorted(glob.glob(os.path.join(dir_path, "emb_ep*_b*_tok.npz")))


def _sample_tokens_from_disk(split_dir: str, sample_count: int, seed: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    files = _list_npz_files(split_dir)
    taken = 0
    token_chunks: List[np.ndarray] = []
    patient_chunks: List[np.ndarray] = []
    tile_chunks: List[np.ndarray] = []
    token_idx_chunks: List[np.ndarray] = []
    epoch_chunks: List[np.ndarray] = []
    files = list(files)
    rng.shuffle(files)  # shuffle for near-uniform coverage
    for file_path in files:
        if taken >= sample_count:
            break
        data = np.load(file_path, allow_pickle=False)
        emb = data["embeddings"]  # (B,T,D)
        batch_size, tokens_per, feat_dim = emb.shape
        tokens_in_file = batch_size * tokens_per
        to_take = min(sample_count - taken, tokens_in_file)
        flat = emb.reshape(tokens_in_file, feat_dim)
        sel = rng.choice(tokens_in_file, size=to_take, replace=False)  # uniform within file
        token_chunks.append(flat[sel])
        p = data["patients"]  # (B,)
        t = data["tiles"]  # (B,)
        e = int(data["epoch"])
        p_rep = np.repeat(p, tokens_per)
        t_rep = np.repeat(t, tokens_per)
        tok_idx = np.tile(np.arange(tokens_per, dtype=np.int32), batch_size)
        patient_chunks.append(p_rep[sel])
        tile_chunks.append(t_rep[sel])
        token_idx_chunks.append(tok_idx[sel])
        epoch_chunks.append(np.full((to_take,), e, dtype=np.int32))
        taken += to_take
    if taken == 0:
        raise RuntimeError(f"no tokens sampled from {split_dir}")
    tokens = np.concatenate(token_chunks, axis=0).astype(np.float32, copy=False)  # fp32 for stability
    meta = {
        "patients": np.concatenate(patient_chunks, axis=0),
        "tiles": np.concatenate(tile_chunks, axis=0),
        "token_idx": np.concatenate(token_idx_chunks, axis=0),
        "epoch": np.concatenate(epoch_chunks, axis=0),
    }
    return tokens, meta


def _zscore(tokens: np.ndarray, mean_vec: np.ndarray, std_vec: np.ndarray) -> np.ndarray:
    eps = 1e-8  # 1e-8 avoids div-by-zero
    return (tokens - mean_vec[None, :]) / np.maximum(std_vec[None, :], eps)


def _pca_reduce(tokens: np.ndarray, k: int) -> np.ndarray:
    centered = tokens - tokens.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    Vtk = Vt[:k, :]
    return centered @ Vtk.T


def _umap2d(tokens: np.ndarray, n_neighbors: int, min_dist: float, seed: int) -> np.ndarray:
    umap_model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=seed, metric="euclidean")
    return umap_model.fit_transform(tokens)


def _plot_scatter(coords: np.ndarray, labels: np.ndarray, *, title: str, size_px: int) -> plt.Figure:
    width_in = size_px / 100.0
    fig = plt.figure(figsize=(width_in, width_in), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=2, cmap="tab20", linewidths=0, alpha=0.9)  # s=2 reduces overplot
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _wandb_log(cfg: Dict[str, Any], coords: np.ndarray, splits_arr: np.ndarray, patients_arr: np.ndarray, counts: Dict[str, int]) -> None:
    run = wandb.init(
        project=cfg["project"],
        group=cfg["group"],
        name=cfg["run_name"],
        tags=cfg.get("tags"),
        config={
            "encoder_type": cfg["encoder_type"],
            "image_size": int(cfg["image_size"]),
            "drop_cls_token": bool(cfg["drop_cls_token"]),
            "embedding_epochs": int(cfg["embedding_epochs"]),
            "umap_sample_per_split": int(cfg["wandb_umap_sample_per_split"]),
            "pca_components": int(cfg["wandb_pca_components"]),
            "umap_neighbors": int(cfg["wandb_umap_neighbors"]),
            "umap_min_dist": float(cfg["wandb_umap_min_dist"]),
        },
    )
    size = int(cfg["wandb_plot_size"])
    fig_split = _plot_scatter(coords, splits_arr, title="UMAP by split", size_px=size)
    fig_patient = _plot_scatter(coords, patients_arr, title="UMAP by patient", size_px=size)
    run.log({
        "umap_split": wandb.Image(fig_split),
        "umap_patient": wandb.Image(fig_patient),
        "token_counts": counts,
    })
    plt.close(fig_split); plt.close(fig_patient)
    run.finish()


def main() -> None:
    args = parse_args()
    raw_cfg = parse_yaml_config(args.config)

    runtime_cfg = _build_runtime_cfg(raw_cfg)

    setup_dump_env()
    cfg = prepare_embeddings_config(runtime_cfg)

    device = resolve_device()
    if device.type not in ("cuda", "mps"):
        raise RuntimeError("accelerator required: cuda or mps")

    encoder = build_encoder(cfg["encoder_type"], device)
    image_size = int(cfg["image_size"])
    epochs_i = int(cfg["embedding_epochs"])
    drop_cls_token = bool(cfg.get("drop_cls_token", True))

    idx_map: Dict[str, Dict[str, Any]] = {}
    stats_accum: Dict[str, Any] = {}
    patient_split_guard: Dict[str, str] = {}

    for split in ("train", "val", "test"):
        export_split_embeddings(
            split=split,
            cfg=cfg,
            encoder=encoder,
            device=device,
            drop_cls_token=drop_cls_token,
            epochs=epochs_i,
            image_size=image_size,
            idx_map=idx_map,
            stats_accum=stats_accum,
            patient_split_guard=patient_split_guard,
        )

    _save_index(cfg["embeddings_path"], idx_map)
    _save_stats(cfg["embeddings_path"], stats_accum, drop_cls_token=drop_cls_token, image_size=image_size)

    if cfg.get("log_to_wandb", False):
        sample_n = int(cfg["wandb_umap_sample_per_split"])
        seed = int(cfg["wandb_umap_seed"])
        split_label_chunks = []
        patient_label_chunks = []
        token_chunks = []
        counts: Dict[str, int] = {}
        for split in ("train", "val", "test"):
            split_dir = os.path.join(cfg["embeddings_path"], split)
            tokens, meta = _sample_tokens_from_disk(split_dir, sample_n, seed)
            token_chunks.append(tokens)
            patient_label_chunks.append(meta["patients"])
            split_label_chunks.append(np.full((tokens.shape[0],), {"train": 0, "val": 1, "test": 2}[split], dtype=np.int32))
            counts[split] = int(tokens.shape[0])

        all_tokens = np.concatenate(token_chunks, axis=0)
        splits_arr = np.concatenate(split_label_chunks, axis=0)
        patients_arr = np.concatenate(patient_label_chunks, axis=0)

        total_count = int(stats_accum["count"])
        mean_vec = (stats_accum["sum"] / total_count).astype(np.float32)
        var_vec = (stats_accum["sumsq"] / total_count) - (mean_vec.astype(np.float64) ** 2)
        var_vec = np.maximum(var_vec, 0.0)
        std_vec = np.sqrt(var_vec).astype(np.float32)

        norm_tokens = _zscore(all_tokens, mean_vec, std_vec)
        pca_tokens = _pca_reduce(norm_tokens, int(cfg["wandb_pca_components"]))
        coords = _umap2d(pca_tokens, int(cfg["wandb_umap_neighbors"]), float(cfg["wandb_umap_min_dist"]), int(cfg["wandb_umap_seed"]))
        _wandb_log(cfg, coords, splits_arr, patients_arr, counts)


if __name__ == "__main__":
    main()
