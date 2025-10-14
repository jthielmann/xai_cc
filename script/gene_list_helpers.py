import glob
import os
from typing import Dict, Any, Optional, List
import pandas as pd


def get_full_gene_list(cfg: Dict[str, Any]) -> List[str]:
    # Validate mutually exclusive CSV configuration early
    if cfg.get("single_csv_path") and (
        cfg.get("train_csv_path") or cfg.get("val_csv_path") or cfg.get("test_csv_path")
    ):
        raise ValueError(
            "Provide either 'single_csv_path' or split-specific CSVs ('train_csv_path'/'val_csv_path'/'test_csv_path'), not both."
        )
    data_dir = cfg.get("data_dir")
    meta_dir = str(cfg.get("meta_data_dir", "meta_data")).strip("/")
    fname = cfg.get("gene_data_filename")

    files = []
    scp = cfg.get("single_csv_path")
    if scp:
        # Resolve relative to data_dir if needed
        path = None
        if not os.path.isabs(scp) and cfg.get("data_dir"):
            cand = os.path.join(cfg["data_dir"], scp)
            if os.path.isfile(cand):
                path = cand
        if path is None and os.path.isfile(scp):
            path = scp
        if path is None:
            raise FileNotFoundError(f"single_csv_path not found: '{scp}'. Tried: '{scp}' and data_dir-joined '{os.path.join(cfg.get('data_dir',''), scp)}'")
        files = [path]
    elif cfg.get("train_csv_path") and os.path.isfile(cfg["train_csv_path"]):
        files = [cfg["train_csv_path"]]
    else:
        sample_ids = cfg.get("sample_ids")
        if sample_ids:
            for sid in sample_ids:
                path = os.path.join(data_dir, sid, meta_dir, fname)
                if os.path.isfile(path):
                    files.append(path)
        else:
            files = glob.glob(os.path.join(data_dir, "*", meta_dir, fname))
            files.sort()

    if not files:
        raise FileNotFoundError(f"No gene data files found under {data_dir}/*/{meta_dir}/{fname}")

    order = None
    inter = None
    for i, path in enumerate(files):
        df = pd.read_csv(path, nrows=1)
        cols = [c for c in df.columns if c != "tile" and not str(c).endswith("_lds_w") and pd.api.types.is_numeric_dtype(df[c])]
        s = set(cols)
        if i == 0:
            order = cols[:]
            inter = s
        else:
            inter &= s
        if not inter:
            break

    genes = [c for c in order if c in inter] if order else []
    if not genes:
        raise ValueError("Could not infer any gene columns. Check your CSV headers and dtypes.")
    return genes

def prepare_gene_list(cfg: Dict[str, Any]) -> List[List[str]]:
    genes = get_full_gene_list(cfg)
    if cfg.get("split_genes_by"):
        k = int(cfg["split_genes_by"])
        if k <= 0:
            raise ValueError("split_genes_by must be a positive integer.")
        chunks = [genes[i:i+k] for i in range(0, len(genes), k)]
        cfg["genes"] = chunks
        cfg["gene_chunks"] = chunks
    else:
        cfg["genes"] = genes
        cfg["gene_chunks"] = [genes]

    cfg["n_gene_chunks"] = len(cfg["gene_chunks"])
    return cfg["genes"]


def get_active_chunk_idx(cfg: Dict[str, Any], chunk: Optional[List[str]] = None) -> int:
    chunks = cfg.get("gene_chunks") or []
    if not chunks:
        return 0
    if chunk is None:
        if cfg.get("genes") and isinstance(cfg["genes"][0], list):
            idx = int(cfg.get("gene_list_index", 1)) - 1
            return max(0, min(idx, len(chunks) - 1))
        # if cfg["genes"] is already a single chunk
        chunk = cfg.get("genes")
    try:
        return next(i for i, ch in enumerate(chunks) if ch == chunk)
    except StopIteration:
        raise RuntimeError(f"chunk not found {chunk} in {chunks}")

def had_split_genes(cfg_dict: Dict[str, Any]) -> bool:
    # check top-level
    if "split_genes_by" in cfg_dict and cfg_dict["split_genes_by"] is not None:
        try:
            return int(cfg_dict["split_genes_by"]) > 0
        except Exception:
            return False
    # check parameters.value
    p = cfg_dict.get("parameters", {}).get("split_genes_by")
    if isinstance(p, dict) and "value" in p and p["value"] is not None:
        try:
            return int(p["value"]) > 0
        except Exception:
            return False
    return False