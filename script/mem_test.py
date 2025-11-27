import argparse, json, os, yaml
from typing import Dict, Any

import pandas as pd
import torch

# project imports
from script.configs.dataset_config import get_dataset_cfg
from script.gene_list_helpers import _is_meta_gene_column
from script.model.model_factory import get_encoder, infer_encoder_out_dim

def _bytes2gb(x: int) -> float: return x / (1024**3)

def _flatten_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten YAML 'parameters: {k:{value:..}}' into a simple cfg dict."""
    cfg = {k: v for k, v in raw.items() if k != "parameters"}
    params = raw.get("parameters", {})
    for k, v in params.items():
        if isinstance(v, dict) and "value" in v:
            cfg[k] = v["value"]
    return cfg

def _count_params(module: torch.nn.Module, trainable_only=True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())

def estimate_heads_params(cfg: Dict[str, Any], D: int) -> int:
    """Per your current design: per-gene modules.
       one_linear_out_layer=False → two-layer per gene: D*M + 2*M + 1
       one_linear_out_layer=True  → single layer per gene: D + 1
    """
    G = len(cfg["genes"])
    if cfg.get("one_linear_out_layer", False):
        per_gene = D + 1
    else:
        M = int(cfg.get("middle_layer_features", 128))
        per_gene = D*M + 2*M + 1
    return G * per_gene

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--optimizer", default="adamw", help="adamw or sgd (affects bytes/param)")
    args = ap.parse_args()

    # 1) load yaml and flatten
    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)
    cfg = _flatten_params(raw)

    # 2) add dataset-derived fields (genes, etc.)
    cfg.update(get_dataset_cfg(cfg))
    if cfg.get("genes", None) is None:
        debug = cfg["debug"]
        meta_data_dir = "/meta_data/"
        patients = cfg["train_samples"] + cfg["val_samples"]
        test_samples = cfg.get("test_samples", None)
        if test_samples:
            patients += test_samples
        data_dir = cfg["data_dir"]
        # Directly resolve metadata CSV using eval override with model_config fallback
        gene_data_filename = cfg.get("gene_data_filename") or cfg.get("model_config", {}).get("gene_data_filename", "gene_data.csv")
        fp = os.path.join(data_dir, patients[0], meta_data_dir.lstrip("/"), gene_data_filename)
        df = pd.read_csv(fp, nrows=1)
        candidates = [
            c for c in df.columns
            if not _is_meta_gene_column(c)
            and not str(c).endswith("_lds_w")
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not candidates:
            raise ValueError("Could not infer gene columns from dataset")
        genes = set(candidates)
        if debug:
            len_genes = len(genes)
            print(f"genes found p0: {len_genes}")
        for idx, patient in enumerate(patients[1:]):
            fp = os.path.join(data_dir, patient, meta_data_dir.lstrip("/"), gene_data_filename)
            df = pd.read_csv(fp, nrows=1)
            genes &= set(df.columns)
            if debug:
                print(f"genes dropped after p{idx + 1}: {len_genes-len(genes)}")
                len_genes = len(genes)
        cfg["genes"] = [c for c in candidates if c in genes]
    # 3) encoder + dims
    enc = get_encoder(cfg["encoder_type"])
    D = infer_encoder_out_dim(enc)

    # 4) trainable params = encoder(if not frozen) + heads
    freeze = bool(cfg.get("freeze_encoder", False))
    if freeze:
        for p in enc.parameters(): p.requires_grad = False
    enc_params = _count_params(enc, trainable_only=True)
    head_params = estimate_heads_params(cfg, D)

    # (Optional SAE note)
    sae_params = 0
    if cfg.get("sae", False):
        print("Note: SAE params not included (set cfg['sae']=False here or add its count if needed).")

    total_trainable = enc_params + head_params + sae_params

    # 5) memory estimate (states = weights+grads+optimizer)
    per_param = 16 if args.optimizer.lower() in {"adam", "adamw"} else 8
    state_bytes = total_trainable * per_param

    # 6) output
    print("\n=== VRAM estimate (states only) ===")
    print(f"encoder_type: {cfg['encoder_type']}, D={D}, genes={len(cfg['genes'])}, "
          f"one_linear_out_layer={cfg.get('one_linear_out_layer', False)}, "
          f"freeze_encoder={freeze}")
    print(f"trainable params total: {total_trainable:,} "
          f"(encoder: {enc_params:,}, heads: {head_params:,})")
    print(f"optimizer: {args.optimizer.lower()}  →  ~{per_param} bytes/param")
    print(f"EST. memory for weights+grads+optimizer: ~{_bytes2gb(state_bytes):.2f} GB\n")

    # quick batch table (states dominate; activations omitted for brevity)
    bs_list = [1, 2, 4, 8, 16, 32]
    print("Batch → estimated GB (states only; activations not included):")
    for B in bs_list:
        print(f"  {B:>2} → {_bytes2gb(state_bytes):.2f} GB  (add ~10–20% headroom)")

if __name__ == "__main__":
    main()
