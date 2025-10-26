import os
import sys
from typing import Any, Dict

# Ensure relative imports work when running from script/
sys.path.insert(0, "..")

# Keep thread settings conservative and avoid Numba OpenMP clashes
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import json, numpy as np
from script.main_utils import parse_args, parse_yaml_config, read_config_parameter, setup_dump_env
from script.embeddings_main_helper import EMBEDDINGS_DTYPE_STR, prepare_embeddings_config, resolve_device, build_encoder, export_split_embeddings


def main() -> None:
    args = parse_args()
    raw_cfg = parse_yaml_config(args.config)

    cfg: Dict[str, Any] = {}
    for key in ("dataset", "debug", "gene_data_filename", "meta_data_dir", "image_size", "batch_size",
                "num_workers", "encoder_type", "embeddings_path", "embeddings_dir", "train_samples",
                "val_samples", "test_samples", "drop_cls_token", "embedding_epochs", "epochs"):
        val = read_config_parameter(raw_cfg, key)
        if val is not None:
            cfg[key] = val

    setup_dump_env(); cfg = prepare_embeddings_config(cfg)

    device = resolve_device()
    image_size = int(cfg.get("image_size", 224))
    encoder = build_encoder(cfg.get("encoder_type"), device)

    drop_cls_token = bool(cfg.get("drop_cls_token", True))
    epochs = int(cfg.get("embedding_epochs", cfg.get("epochs", 1) or 1))

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
            epochs=epochs,
            image_size=image_size,
            idx_map=idx_map,
            stats_accum=stats_accum,
            patient_split_guard=patient_split_guard,
        )

    with open(os.path.join(cfg["embeddings_path"], "index.json"), "w") as f:
        json.dump(idx_map, f)

    if stats_accum.get("count", 0) > 0:
        count = int(stats_accum["count"])
        sumv = stats_accum["sum"]
        sumsq = stats_accum["sumsq"]
        mean = (sumv / count).astype(np.float32)
        var = (sumsq / count) - (mean.astype(np.float64) ** 2)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var).astype(np.float32)
        stats_payload = {
            "dtype": EMBEDDINGS_DTYPE_STR,
            "drop_cls_token": drop_cls_token,
            "image_size": image_size,
            "feature_dim": int(mean.shape[0]),
            "count": count,
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
        with open(os.path.join(cfg["embeddings_path"], "embed_stats.json"), "w") as f:
            json.dump(stats_payload, f)


if __name__ == "__main__":
    main()
