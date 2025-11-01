from __future__ import annotations

import random
import torch.nn.functional
from torchvision import transforms
from script.data_processing.image_transforms import get_eval_transforms, get_transforms
from script.configs.dataset_config import get_dataset_cfg
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union
DEFAULT_RANDOM_SEED = 42

import matplotlib.pyplot as plt
from torchvision import transforms as T
import os

from pathlib import Path
import json
import numpy as np
import pandas as pd


def _is_unnamed_column(name: Any) -> bool:
    try:
        s = str(name).strip().lower()
    except Exception:
        s = str(name)
    return s.startswith("unnamed") or s == ""


def seed_basic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def seed_torch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_everything(seed=DEFAULT_RANDOM_SEED):
    #seed_basic(seed)
    seed_torch(seed)


# Avoid seeding at import time unless explicitly requested via env var
if os.environ.get("ST_SEED_DATA_LOADER_ON_IMPORT", "0") == "1":
    seed_everything(seed=DEFAULT_RANDOM_SEED)


def log_training(date, training_log):
    with open(training_log, "a") as f:
        f.write(
            date + " Resnet50 - single gene\top: AdamW\telrs: 0.9\tlfn: MSE Loss\n")  # Adapt to model and gene name(s) getting trained


import torch
from torch.utils.data import Dataset
from PIL import Image
class STDataset(Dataset):
    def __init__(
        self,
        df,
        *,
        image_transforms=None,
        inputs_only: bool = False,
        genes: list[str] | None = None,
        use_weights: bool = False,
        return_floats: bool = False,
        gene_list_index = -1,
        split_genes_by = 100,
        return_patient_and_tilepath = True
    ):
        self.df = df.reset_index(drop=True)
        self.transforms = image_transforms
        self.inputs_only = inputs_only
        self.return_floats = return_floats
        self.return_patient_and_tilepath = return_patient_and_tilepath

        # Determine gene columns explicitly (stable order)
        if genes is None:
            # auto-detect genes = all numeric columns except 'tile' and *_lds_w
            candidates = []
            for c in self.df.columns:
                if c == "tile" or c == "patient" or str(c).endswith("_lds_w"):
                    continue
                if _is_unnamed_column(c):
                    continue
                # keep only numeric columns
                if pd.api.types.is_numeric_dtype(self.df[c]):
                    candidates.append(c)

            if not candidates:
                raise ValueError("Could not infer gene columns; please pass `genes`.")

            if gene_list_index and gene_list_index > 0:
                if split_genes_by is not None and split_genes_by > 0:
                    k = int(split_genes_by)
                    if k <= 0:
                        raise ValueError("split_genes_by must be a positive integer.")
                    chunks = [candidates[i:i + k] for i in range(0, len(candidates), k)]
                    if gene_list_index >= len(chunks):
                        raise IndexError(f"gene_list_index={gene_list_index} exceeds number of chunks={len(chunks)}.")
                    candidates = chunks[gene_list_index]
            self.genes = list(candidates)
        else:
            # Sanitize provided list: drop any accidental Unnamed/blank entries
            genes = [g for g in genes if not _is_unnamed_column(g)]
            if not genes:
                raise ValueError("No valid gene columns after filtering out unnamed/blank entries.")
            missing = [g for g in genes if g not in self.df.columns]
            if missing:
                raise ValueError(f"Genes missing in DataFrame: {missing}")
            self.genes = list(genes)

        # Optional per-gene weights: try to read a vector w[g] per sample
        self.use_weights = use_weights
        if self.use_weights:
            # For each gene, we expect a column f"{gene}_lds_w"
            missing_w = [g for g in self.genes if f"{g}_lds_w" not in self.df.columns]
            if missing_w:
                raise ValueError(
                    f"Weight columns not found for genes: {missing_w}. "
                    "Did you generate LDS weights or set lds_weight_dir?"
                )

    def __len__(self):
        return len(self.df)

    def get_tilename(self, index: int) -> str:
        return self.df.iloc[index]["tile"]

    def get_patient(self, index: int) -> str:
        return self.df.iloc[index]["patient"]

    def _load_image(self, path: str):
        img = Image.open(path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img

    def _row_to_target(self, row) -> torch.Tensor:
        # Always return shape (G,), even if G == 1
        vals = [float(row[g]) for g in self.genes]
        return torch.tensor(vals, dtype=torch.float32)

    def _row_to_weights(self, row) -> torch.Tensor:
        # Always return shape (G,), even if G == 1
        w_vals = [float(row[f"{g}_lds_w"]) for g in self.genes]
        return torch.tensor(w_vals, dtype=torch.float32)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = self._load_image(row["tile"])

        if self.inputs_only:
            return img

        if self.return_floats:
            y = np.array([float(row[g]) for g in self.genes], dtype=np.float32)   # shape (G,)
            if self.use_weights:
                w = np.array([float(row[f"{g}_lds_w"]) for g in self.genes], dtype=np.float32)  # shape (G,)
                return img, y, w
            return img, y

        # default tensor path
        y = self._row_to_target(row)  # torch.float32, shape (G,)
        if self.use_weights:
            w = self._row_to_weights(row)  # torch.float32, shape (G,)
            return img, y, w

        if self.return_patient_and_tilepath:
            return img, y, self.get_patient(index), self.get_tilename(index)
        else:
            return img, y


class STDatasetUMAP(STDataset):
    def __init__(self, df, *, image_transforms=None):
        super().__init__(
            df,
            image_transforms=image_transforms,
            inputs_only=True, # umap
            genes=[],
            use_weights=False,
            return_floats=False
        )

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = self._load_image(row["tile"])
        return img, row["tile"]


class TileLoader:
    def __init__(self):
        # Use centralized ImageNet-normalized eval transforms
        self.transforms = get_eval_transforms(image_size=224)

    def open(self, path):
        a = Image.open(path).convert("RGB")
        return self.transforms(a)


class Subset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    # get image and label tensor from dataset and transform
    def __getitem__(self, index):
        a, gene_vals = self.subset[index]
        if self.transform:
            a = self.transform(a)
        return a, gene_vals

    # get length of dataset
    def __len__(self):
        return len(self.subset)


def get_patient_loader(data_dir, patient, genes=None):

    if genes is None:
        columns_of_interest = ["tile", "RUBCNL"]
    else:
        columns_of_interest = ["tile"]
        for gene in genes:
            columns_of_interest.append(gene)
    train_st_dataset = pd.DataFrame(columns=columns_of_interest)

    # generate training dataframe with all training samples
    st_dataset = pd.read_csv(data_dir + patient + "/meta_data/gene_data.csv", index_col=-1)
    st_dataset["tile"] = st_dataset.index
    st_dataset['tile'] = st_dataset['tile'].apply(lambda x: str(data_dir) + "/" + str(patient) + "/tiles/" + str(x))

    if train_st_dataset.empty:
        train_st_dataset = st_dataset[columns_of_interest]
    else:
        train_st_dataset = pd.concat([train_st_dataset, st_dataset[columns_of_interest]])  # concat all samples

    train_st_dataset.reset_index(drop=True, inplace=True)
    loaded_train_dataset = STDataset(train_st_dataset)
    return loaded_train_dataset

import torch, math

def make_weights(
    raw: Dict[str, torch.Tensor],
    *,
    clip_max: float = 10.0,
    r1: float = 20.0,
    r2: float = 100.0,
) -> Dict[str, torch.Tensor]:
    scaled = {}
    for g, w in raw.items():
        w = w.float()
        ratio = (w.max() / w.mean()).item()
        if ratio <= r1:
            alpha = 1.0
        elif ratio >= r2:
            alpha = 0.0
        else:
            t = (math.log10(ratio) - math.log10(r1)) / (math.log10(r2) - math.log10(r1))
            alpha = 1.0 - t
        w2 = torch.log1p(w) if alpha == 0.0 else (w / w.mean()) ** alpha
        # Normalize, clamp, then re-normalize to keep mean ~ 1
        eps = 1e-12
        w2 = w2 / torch.clamp(w2.mean(), min=eps)
        w2 = torch.clamp(w2, max=clip_max)
        w2 = w2 / torch.clamp(w2.mean(), min=eps)
        scaled[g] = w2.to(dtype=torch.float32)
    return scaled


def _compute_bin_edges(
    values: np.ndarray,
    num_bins: int,
    *,
    strategy: str = "quantile",
    clip: float = 0.0,
) -> np.ndarray:
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    if values.size == 0:
        raise ValueError("Cannot compute bin edges for empty value array.")

    strategy = str(strategy).lower()
    clip = float(clip)
    if clip < 0.0 or clip >= 0.5:
        raise ValueError("clip must be in the interval [0, 0.5).")

    v_min = float(np.min(values))
    v_max = float(np.max(values))
    if not np.isfinite(v_min) or not np.isfinite(v_max):
        raise ValueError("Values contain non-finite entries.")
    if v_min == v_max:
        return np.linspace(v_min, v_min + 1e-6, num_bins + 1, dtype=np.float32)

    if strategy == "quantile":
        qs = np.linspace(0.0, 1.0, num_bins + 1)
        if clip > 0.0:
            qs = np.clip(qs, clip, 1.0 - clip)
            qs[0] = 0.0
            qs[-1] = 1.0
        edges = np.quantile(values, qs)
    elif strategy == "linspace":
        edges = np.linspace(v_min, v_max, num_bins + 1)
    else:
        raise ValueError(f"Unknown bin edge strategy '{strategy}'. Expected 'quantile' or 'linspace'.")

    edges = np.asarray(edges, dtype=np.float32)
    edges[0] = v_min
    edges[-1] = v_max

    diffs = np.diff(edges)
    if np.any(diffs <= 0):
        # Fallback to evenly spaced bins if quantiles collapse due to repeated values
        edges = np.linspace(v_min, v_max, num_bins + 1, dtype=np.float32)

    return edges



def get_all_genes():
    pass

# contains tile path and gene data
def get_base_dataset(
    data_dir: str | Path,
    samples: List[str],
    genes: list[str] | None = None,
    *,
    max_len: int | None = None,
    bins: int = 1,
    gene_data_filename: str = "gene_data.csv",
    meta_data_dir: str = "/meta_data/",
    lds_smoothing_csv: str | Path | None = None,      # <─ NOW one file, not a dir
    weight_transform: str = "inverse",
    weight_clamp: int = 10,
    lds_bin_edge_strategy: str = "quantile",
    lds_bin_edge_clip: float = 0.0,
    precomputed_edges: Dict[str, np.ndarray] | None = None,
    return_edges: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, np.ndarray]]]:
    if lds_smoothing_csv is not None and genes is None:
        raise RuntimeError(f"lds_smoothing_csv is not None ({lds_smoothing_csv}) and genes is None")
    # normalize meta data directory token (support 'meta_data', '/meta_data', 'meta_data/')
    meta_dir = meta_data_dir.strip("/")
    # Safety: never include unnamed/index columns as genes; sanitize early
    if genes is not None:
        genes = [g for g in genes if not _is_unnamed_column(g)]
        if not genes:
            raise ValueError("Provided gene list contains only unnamed/blank columns; nothing to use.")
    # Always try to include spatial coordinates if present ('x','y'), even when specific genes are requested.
    # We will filter by available columns per-file to avoid read_csv errors.
    columns_of_interest = ["tile"] + genes if genes else None
    dfs = []

    for patient in samples:
        fp = os.path.join(data_dir, patient, meta_dir, gene_data_filename)
        if columns_of_interest is None:
            # No explicit gene list: read all columns and let downstream infer gene columns
            df = pd.read_csv(fp, nrows=max_len)
        else:
            # Best bet: request gene columns plus spatial coords 'x' and 'y' directly.
            wanted = list(columns_of_interest)
            for extra in ("x", "y"):
                if extra not in wanted:
                    wanted.append(extra)
            df = pd.read_csv(fp, usecols=wanted, nrows=max_len)
        df["tile"] = df["tile"].apply(lambda t: os.path.join(data_dir, patient, "tiles", t))
        df["patient"] = patient
        dfs.append(df)

    base_df = pd.concat(dfs, ignore_index=True)

    # Safety (again): ensure genes are sanitized
    if genes is not None:
        genes = [g for g in genes if not _is_unnamed_column(g)]

    edges_lookup = precomputed_edges
    edges_result: Dict[str, np.ndarray] = dict(edges_lookup) if return_edges else edges_lookup

    if lds_smoothing_csv is not None:
        lgw_res = load_gene_weights(
            lds_smoothing_csv, genes=genes, weight_transform=weight_transform, return_scales=True
        )
        gene2weights, calib_by_gene = lgw_res
        gene2weights = make_weights(gene2weights, clip_max=weight_clamp, r1=20, r2=100)
        # Apply per-gene calibration scale AFTER make_weights normalization
        for g in list(gene2weights.keys()):
            scale = float(calib_by_gene.get(g, 1.0))
            if scale != 1.0:
                gene2weights[g] = gene2weights[g] * scale

        # Collect weight columns first to avoid DataFrame fragmentation from many inserts
        weight_cols: Dict[str, np.ndarray] = {}
        # Ensure all requested (sanitized) genes exist in the DataFrame and have weights
        valid_genes = []
        for g in genes:
            if g not in base_df.columns:
                raise ValueError(f"Gene column missing from DataFrame: {g}")
            if g not in gene2weights:
                raise ValueError(f"No weights for {g}")
            valid_genes.append(g)
        for g in valid_genes:
            w_vec = gene2weights[g]  # (K,)
            K = len(w_vec)

            vals = base_df[g].to_numpy()
            if g in edges_lookup:
                edges = np.asarray(edges_lookup[g], dtype=np.float32)
            else:
                edges = _compute_bin_edges(
                    vals,
                    K,
                    strategy=lds_bin_edge_strategy,
                    clip=lds_bin_edge_clip,
                )
                if return_edges:
                    edges_result[g] = edges
            if edges.shape[0] != K + 1:
                raise ValueError(f"Edges for gene '{g}' must have length {K + 1}, got {edges.shape[0]}.")
            idx = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, K - 1)
            weight_cols[f"{g}_lds_w"] = w_vec[idx]

        if weight_cols:
            wdf = pd.DataFrame(weight_cols, index=base_df.index)
            base_df = pd.concat([base_df, wdf], axis=1)

    if bins > 1 and genes:
        gene_values = base_df[genes[0]]
        gene_bins   = pd.cut(gene_values, bins)
        groups      = base_df.groupby(gene_bins)
        sample_size = groups.size().max()
        base_df     = (groups
                       .apply(lambda x: x.sample(sample_size, replace=True) if len(x) else x)
                       .reset_index(drop=True))

    if return_edges:
        return base_df, edges_result
    return base_df


def plot_gene_histograms_per_patient(
    df,
    genes,
    patients=None,
    bins=50,
    density=False,
    log=False,
    save_dir=None,
    show=True
):
    if patients is None:
        patients = sorted(df["patient"].unique())

    out_paths = []

    for g in genes:
        for p in patients:
            vals = df.loc[df["patient"] == p, g].dropna().to_numpy()
            if vals.size == 0:
                continue

            plt.figure()
            plt.hist(vals, bins=bins, density=density, log=log)
            plt.title(f"Histogram — patient={p} · gene={g}")
            plt.xlabel(g)
            plt.ylabel("density" if density else "count")

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                out = os.path.join(save_dir, f"hist_{g}_patient_{p}.png")
                plt.savefig(out, dpi=150, bbox_inches="tight")
                out_paths.append(out)

            if show:
                plt.show()

            plt.close()

    return out_paths

def overlay_patient_hists_for_gene(df, gene, patients=None, bins=50, density=True, log=False):
    if patients is None:
        patients = sorted(df["patient"].unique())
    plt.figure()
    for p in patients:
        vals = df.loc[df["patient"] == p, gene].dropna().to_numpy()
        if vals.size == 0:
            continue
        plt.hist(vals, bins=bins, alpha=0.4, density=density, log=log)  # alpha helps visibility
    plt.title(f"Overlayed histograms — {gene}")
    plt.xlabel(gene); plt.ylabel("density" if density else "count")
    plt.show(); plt.close()

def get_dataset(
    data_dir: str | Path,
    genes: list[str] | None = None,
    *,
    transforms=None,
    samples: list[str] | None = None,
    max_len: int | None = None,
    bins: int = 1,
    only_inputs: bool = False,
    gene_data_filename: str = "gene_data.csv",
    meta_data_dir: str = "/meta_data/",
    lds_smoothing_csv: str | Path | None = None,
    weight_transform: str = "inverse",
    weight_clamp: int = 10,
    return_floats: bool = False,
    lds_bin_edge_strategy: str = "quantile",
    lds_bin_edge_clip: float = 0.0,
    precomputed_bin_edges: Dict[str, np.ndarray] | None = None,
    return_edges: bool = False,
    return_patient_and_tilepath=False
):
    print(data_dir)
    if samples is None:
        samples = [f.name for f in os.scandir(data_dir) if f.is_dir() and not f.name.startswith(".") and not f.name.startswith("_")]

    base_result = get_base_dataset(
        data_dir=data_dir,
        samples=samples,
        genes=genes,
        max_len=max_len,
        bins=bins,
        gene_data_filename=gene_data_filename,
        meta_data_dir=meta_data_dir,
        lds_smoothing_csv=lds_smoothing_csv,
        weight_transform=weight_transform,
        weight_clamp=weight_clamp,
        lds_bin_edge_strategy=lds_bin_edge_strategy,
        lds_bin_edge_clip=lds_bin_edge_clip,
        precomputed_edges=precomputed_bin_edges,
        return_edges=return_edges,
    )
    if return_edges:
        df, edges = base_result
    else:
        df = base_result
        edges = precomputed_bin_edges

    # create the PyTorch-compatible dataset
    ds = STDataset(
        df,
        image_transforms=transforms,
        inputs_only=only_inputs,
        genes=genes,
        use_weights=lds_smoothing_csv is not None,
        return_floats=return_floats,
        return_patient_and_tilepath=return_patient_and_tilepath
    )
    if return_edges:
        return ds, edges
    return ds


def get_dataset_from_config(
    dataset_name: str,
    genes: list[str] | None = None,
    *,
    split: Literal["train", "val", "test"] = "train",
    debug: bool = False,
    transforms=None,
    samples: list[str] | None = None,
    max_len: int | None = None,
    bins: int = 1,
    only_inputs: bool = False,
    gene_data_filename: str = "gene_data.csv",
    meta_data_dir: str = "/meta_data/",
    lds_smoothing_csv: str | Path | None = None,
    weight_transform: str = "inverse",
    weight_clamp: int = 10,
    return_floats: bool = False,
    lds_bin_edge_strategy: str = "quantile",
    lds_bin_edge_clip: float = 0.0,
    precomputed_bin_edges: Dict[str, np.ndarray] | None = None,
    return_edges: bool = False,
) -> STDataset:

    cfg: dict[str, Any] = {"dataset": dataset_name}
    if debug:
        cfg["debug"] = True

    dataset_cfg = get_dataset_cfg(cfg)
    data_dir = dataset_cfg["data_dir"]

    resolved_samples = samples
    if resolved_samples is None:
        split_key_map = {
            "train": "train_samples",
            "val": "val_samples",
            "test": "test_samples",
        }
        key = split_key_map.get(split)
        if key is None:
            raise ValueError(f"Unsupported split '{split}'. Expected one of: {tuple(split_key_map)}")

        resolved_samples = dataset_cfg.get(key)
        if resolved_samples is None and split == "test":
            resolved_samples = dataset_cfg.get("test_samples_all")

        if resolved_samples is None:
            raise KeyError(
                f"Dataset '{dataset_name}' does not provide samples for split '{split}'. "
                "Pass 'samples' explicitly to override."
            )

    resolved_gene_data_filename = dataset_cfg.get("gene_data_filename", gene_data_filename)
    resolved_meta_data_dir = dataset_cfg.get("meta_data_dir", meta_data_dir)

    return get_dataset(
        data_dir=data_dir,
        genes=genes,
        transforms=transforms,
        samples=resolved_samples,
        max_len=max_len,
        bins=bins,
        only_inputs=only_inputs,
        gene_data_filename=resolved_gene_data_filename,
        meta_data_dir=resolved_meta_data_dir,
        lds_smoothing_csv=lds_smoothing_csv,
        weight_transform=weight_transform,
        weight_clamp=weight_clamp,
        return_floats=return_floats,
        lds_bin_edge_strategy=lds_bin_edge_strategy,
        lds_bin_edge_clip=lds_bin_edge_clip,
        precomputed_bin_edges=precomputed_bin_edges,
        return_edges=return_edges,
    )


# returns imgs and tile paths
def get_dataset_for_umap(data_dir, genes, transforms=None, samples=None, meta_data_dir="/meta_data/", max_len=None, bins=1, only_inputs=False):
    if samples is None:
        samples = [os.path.basename(f) for f in os.scandir(data_dir) if f.is_dir()]
    gene_data_df = get_base_dataset(
        data_dir=data_dir,
        samples=samples,
        genes=genes,
        meta_data_dir=meta_data_dir,
        max_len=max_len,
        bins=bins,
    )
    st_dataset = STDatasetUMAP(gene_data_df, image_transforms=transforms, inputs_only=only_inputs)
    return st_dataset


def get_base_dataset_single_file(
    csv_path: str | Path,
    *,
    data_dir: str | Path | None = None,
    genes: list[str] | None = None,
    max_len: int | None = None,
    bins: int = 1,
    lds_smoothing_csv: str | Path | None = None,
    weight_transform: str = "inverse",
    weight_clamp: int = 10,
    tile_subdir: str | None = None,
    split: str | list[str] | None = None,
    split_col_name: str = "split",
    lds_bin_edge_strategy: str = "quantile",
    lds_bin_edge_clip: float = 0.0,
    precomputed_edges: Dict[str, np.ndarray] | None = None,
    return_edges: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, np.ndarray]]]:
    """Build a base DataFrame from a single top-level CSV.

    The CSV should contain at least a `tile` column and gene columns. If `data_dir`
    is provided and `tile` paths are relative, they are joined with
    `data_dir` (and `tile_subdir` if given).
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, nrows=max_len)

    # Safety: never include unnamed/index-like columns as genes if provided
    if genes is not None:
        genes = [g for g in genes if not _is_unnamed_column(g)]
        if not genes:
            raise ValueError("Provided gene list contains only unnamed/blank columns; nothing to use.")

    # Optional row filtering by split column
    if split is not None:
        if split_col_name not in df.columns:
            raise ValueError(f"Column '{split_col_name}' not found in {csv_path}")
        if isinstance(split, (list, tuple, set)):
            df = df[df[split_col_name].isin(list(split))].copy()
        else:
            df = df[df[split_col_name] == split].copy()

    # Ensure absolute tile paths if data_dir is given and tiles are relative
    if data_dir is not None:
        base = Path(data_dir)
        if tile_subdir:
            base = base / tile_subdir
        df["tile"] = df["tile"].apply(lambda t: str(base / str(t)) if not os.path.isabs(str(t)) else str(t))

    # Ensure a patient column exists (optional)
    if "patient" not in df.columns:
        df["patient"] = "all"

    edges_lookup = precomputed_edges
    edges_result: Dict[str, np.ndarray] = dict(edges_lookup) if return_edges else edges_lookup

    # Optionally attach LDS weights
    if lds_smoothing_csv is not None:
        # Determine gene list if not provided
        if genes is None:
            candidates = []
            for c in df.columns:
                if c == "tile" or str(c).endswith("_lds_w") or c == "patient":
                    continue
                if _is_unnamed_column(c):
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    candidates.append(c)
            genes = list(candidates)

        if genes:
            lgw_res = load_gene_weights(
                lds_smoothing_csv, genes=genes, weight_transform=weight_transform, return_scales=True
            )
            gene2weights, calib_by_gene = lgw_res
            gene2weights = make_weights(gene2weights, clip_max=weight_clamp, r1=20, r2=100)
            # Apply per-gene calibration after normalization
            for g in list(gene2weights.keys()):
                scale = float(calib_by_gene.get(g, 1.0))
                if scale != 1.0:
                    gene2weights[g] = gene2weights[g] * scale

            # Batch-add weight columns to avoid fragmentation
            weight_cols: Dict[str, np.ndarray] = {}
            # Ensure all requested (sanitized) genes exist and have weights
            valid_genes = []
            for g in genes:
                if g not in df.columns:
                    raise ValueError(f"Gene column missing from DataFrame: {g}")
                if g not in gene2weights:
                    raise ValueError(f"No weights for {g}")
                valid_genes.append(g)
            for g in valid_genes:
                w_vec = gene2weights[g]
                K = len(w_vec)
                vals = df[g].to_numpy()
                if g in edges_lookup:
                    edges = np.asarray(edges_lookup[g], dtype=np.float32)
                else:
                    edges = _compute_bin_edges(
                        vals,
                        K,
                        strategy=lds_bin_edge_strategy,
                        clip=lds_bin_edge_clip,
                    )
                    if return_edges:
                        edges_result[g] = edges
                if edges.shape[0] != K + 1:
                    raise ValueError(
                        f"Edges for gene '{g}' must have length {K + 1}, got {edges.shape[0]}."
                    )
                idx = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, K - 1)
                weight_cols[f"{g}_lds_w"] = w_vec[idx]

            if weight_cols:
                wdf = pd.DataFrame(weight_cols, index=df.index)
                df = pd.concat([df, wdf], axis=1)

    # Optional bin oversampling based on the first gene
    if bins > 1 and genes:
        gene_values = df[genes[0]]
        gene_bins = pd.cut(gene_values, bins)
        groups = df.groupby(gene_bins)
        sample_size = groups.size().max()
        df = (
            groups.apply(lambda x: x.sample(sample_size, replace=True) if len(x) else x)
            .reset_index(drop=True)
        )

    if return_edges:
        return df, edges_result
    return df


def get_dataset_single_file(
    csv_path: str | Path,
    *,
    data_dir: str | Path | None = None,
    genes: list[str] | None = None,
    transforms=None,
    max_len: int | None = None,
    bins: int = 1,
    only_inputs: bool = False,
    lds_smoothing_csv: str | Path | None = None,
    weight_transform: str = "inverse",
    weight_clamp: int = 10,
    return_floats: bool = False,
    tile_subdir: str | None = None,
    split: str | list[str] | None = None,
    split_col_name: str = "split",
    lds_bin_edge_strategy: str = "quantile",
    lds_bin_edge_clip: float = 0.0,
    precomputed_bin_edges: Dict[str, np.ndarray] | None = None,
    return_edges: bool = False,
):
    """Return an `STDataset` built from a single top-level CSV.

    Mirrors `get_dataset`, but does not expect per-patient subdirectories.
    """
    base_result = get_base_dataset_single_file(
        csv_path=csv_path,
        data_dir=data_dir,
        genes=genes,
        max_len=max_len,
        bins=bins,
        lds_smoothing_csv=lds_smoothing_csv,
        weight_transform=weight_transform,
        weight_clamp=weight_clamp,
        tile_subdir=tile_subdir,
        split=split,
        split_col_name=split_col_name,
        lds_bin_edge_strategy=lds_bin_edge_strategy,
        lds_bin_edge_clip=lds_bin_edge_clip,
        precomputed_edges=precomputed_bin_edges,
        return_edges=return_edges,
    )
    if return_edges:
        df, edges = base_result
    else:
        df = base_result
        edges = precomputed_bin_edges

    ds = STDataset(
        df,
        image_transforms=transforms,
        inputs_only=only_inputs,
        genes=genes,
        use_weights=lds_smoothing_csv is not None,
        return_floats=return_floats,
    )
    if return_edges:
        return ds, edges
    return ds

def get_dino_dataset(csv_path, dino_transforms=None, max_len=None, bins=1, device_handling=False):
    if dino_transforms is None:
        from script.data_processing.image_transforms import get_transforms as _gt
        dino_transforms = _gt(None, split='train')
    file_df = pd.read_csv(csv_path, nrows=max_len)
    # Ensure STDataset does not attempt to infer gene columns (CSV has none for DINO)
    st_dataset = STDataset(file_df, image_transforms=dino_transforms, inputs_only=True, genes=[])
    return st_dataset


class NCT_CRC_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, classes, use_tiles_sub_dir=False, image_transforms=None, device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu", label_as_string=True):
        self.transforms = image_transforms
        self.device = device
        self.label_as_string = label_as_string
        self.classes = classes
        self.dataframe = pd.DataFrame(columns=["tile", "class"])
        for c in classes:
            class_dir = data_dir + "/" + c
            if use_tiles_sub_dir:
                class_dir += "/tiles/"
            for file in os.scandir(class_dir):
                if file.is_file() and file.name.endswith(".tif"):
                    self.dataframe = pd.concat([self.dataframe, pd.DataFrame({"tile": [file.path], "class": [c if label_as_string else classes.index(c)]})], ignore_index=True)

    def __len__(self):
        return len(self.dataframe)

    def get_tilepath(self, index):
        return self.dataframe.iloc[index]["tile"]

    def get_class(self, index):
        return self.dataframe.iloc[index]["class"]

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        img = Image.open(row["tile"]).convert("RGB")
        if self.transforms:
            img = self.transforms(img)

        return img, row["class"]

def get_label_dataframe(data_dir, samples, meta_data_dir="/meta_data/", max_len=None, gene_data_filename="gene_data.csv"):
    datasets = []  # Use a list to store DataFrames

    # Normalize meta_data_dir to a single path token (no leading/trailing slashes)
    meta_dir = str(meta_data_dir).strip("/")

    for patient in samples:
        # Build path robustly: <data_dir>/<patient>/<meta_dir>/<gene_data_filename>
        file_path = os.path.join(data_dir, patient, meta_dir, gene_data_filename)
        # Read the CSV file, excluding the 'tile' column because it is not needed for label smoothing
        st_dataset_patient = pd.read_csv(file_path, nrows=max_len, usecols=lambda col: col != "tile")
        # Drop accidental index columns like 'Unnamed: 0'
        bad = [c for c in st_dataset_patient.columns if _is_unnamed_column(c)]
        if bad:
            st_dataset_patient = st_dataset_patient.drop(columns=bad)
        datasets.append(st_dataset_patient)
    st_dataset = pd.concat(datasets, ignore_index=True)
    return st_dataset



class label_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, samples, gene_data_filename, gene=None, meta_data_dir="/meta_data/", max_len=None, dtype=torch.float32):
        self.dataframe = get_label_dataframe(data_dir, samples=samples, meta_data_dir=meta_data_dir, max_len=max_len, gene_data_filename=gene_data_filename)
        if gene is not None and gene not in self.dataframe.columns:
            raise ValueError(f"Gene '{gene}' not found in dataframe columns.")
        self.gene = gene
        self.dtype = dtype

    def __len__(self):
        return len(self.dataframe)

    def set_gene(self, gene):
        if gene not in self.dataframe.columns:
            raise ValueError(f"Gene '{gene}' not found in dataframe columns.")
        self.gene = gene
    def __getitem__(self, idx):
        val = self.dataframe.iloc[idx, self.dataframe.columns.get_loc(self.gene)]
        return torch.tensor(float(val), dtype=self.dtype) # supposed to be a 0D tensor

def load_best_smoothing(csv_path: str | Path, gene: str) -> Dict[str, Any]:
    """Load best smoothing params for a gene using JS divergence (smaller is better).

    Expects a CSV produced by `script/data_processing/lds_coad.py:grid_search_lds`, which
    contains at least the columns: gene, bins, kernel_size, sigma, weights_json, js.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    # Drop any rows with unnamed/blank gene labels to avoid accidental matches
    if "gene" in df.columns:
        mask = df["gene"].astype(str).str.strip().str.lower().str.startswith("unnamed")
        df = df.loc[~mask].copy()
    sub = df[df["gene"] == gene]
    if sub.empty:
        raise KeyError(f"Gene {gene!r} not present in {csv_path}")

    if "js" not in sub.columns:
        raise ValueError("CSV does not contain 'js' column for selection.")

    # select the row with minimal JS divergence
    row = sub.loc[sub["js"].idxmin()]

    weights = np.array(json.loads(row["weights_json"]), dtype=float)
    return {
        "bins": int(row["bins"]),
        "kernel_size": int(row["kernel_size"]),
        "sigma": float(row["sigma"]),
        "js": float(row["js"]),
        "weights": weights,
    }


def load_gene_weights(
    csv_path: str | Path,
    genes: List[str],
    *,
    weight_transform: Literal["inverse", "sqrt-inverse", "none"] = "inverse",
    selection_metric: str = "js",
    mode: Literal["min", "max"] = "min",
    eps: float = 1e-12,
    renorm_mean1: bool = True,
    clip_max: Optional[float] = None,
    return_scales: bool = False,
) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, float]]]:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    # Drop any rows with unnamed/blank gene labels to avoid accidental matches
    if "gene" in df.columns:
        mask = df["gene"].astype(str).str.strip().str.lower().str.startswith("unnamed")
        df = df.loc[~mask].copy()

    # restrict to requested genes early
    df = df[df["gene"].isin(genes)].copy()
    if df.empty:
        raise ValueError("No rows found in CSV for the requested genes.")

    if selection_metric not in df.columns:
        raise ValueError(
            f"selection_metric '{selection_metric}' not found in CSV. "
            f"Available: {sorted(df.columns)}"
        )

    # choose best row per gene by the specified metric
    if mode == "min":
        idx = df.groupby("gene")[selection_metric].idxmin()
    elif mode == "max":
        idx = df.groupby("gene")[selection_metric].idxmax()
    else:
        raise ValueError("mode must be 'min' or 'max'.")

    best_rows = df.loc[idx].set_index("gene")

    gene2weights: Dict[str, torch.Tensor] = {}
    calib_by_gene: Dict[str, float] = {}
    for gene in genes:
        if gene not in best_rows.index:
            continue  # skip genes not present in CSV
        row = best_rows.loc[gene]
        p = torch.tensor(json.loads(row["weights_json"]), dtype=torch.float32)

        # Normalize to a proper probability vector (sum=1)
        p_sum = torch.clamp(p.sum(), min=eps)
        p = p / p_sum

        # Transform to weights
        if weight_transform == "inverse":
            w = 1.0 / torch.clamp(p, min=eps)
        elif weight_transform == "sqrt-inverse":
            w = 1.0 / torch.clamp(p, min=eps).sqrt()
        elif weight_transform == "none":
            w = p.clone()
        else:
            raise ValueError(f"Unknown weight_transform: {weight_transform}")

        # Optional clipping to tame tail bins
        if clip_max is not None:
            w = torch.clamp(w, max=float(clip_max))

        # Normalize weights to mean 1 (helps keep loss magnitudes comparable)
        if renorm_mean1:
            w = w / torch.clamp(w.mean(), min=eps)

        gene2weights[gene] = w
        # Optional per-gene calibration scale if present in CSV (e.g., from tune_wmse_weights)
        if "calibration_scale" in best_rows.columns:
            try:
                calib_by_gene[gene] = float(row["calibration_scale"])
            except Exception:
                calib_by_gene[gene] = 1.0
        else:
            calib_by_gene[gene] = 1.0

    if return_scales:
        return gene2weights, calib_by_gene
    return gene2weights


# return_floats -> python float instead of torch
class STSpatialDataset(Dataset):
    def __init__(
        self,
        df,
        *,
        image_transforms=None,
        genes: Optional[List[str]] = None,
        return_floats: bool = False,
        dtype: torch.dtype = torch.float32,
        patient_filter: Optional[Union[str, Sequence[str]]] = None,
        return_patient: bool = False,
    ):
        # Basic column checks
        required_cols = {"tile", "x", "y", "patient"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame must include columns {sorted(required_cols)}; missing {sorted(missing)}.")

        # Optional patient filtering
        if patient_filter is not None:
            if isinstance(patient_filter, str):
                keep = {patient_filter}
            else:
                keep = set(patient_filter)
            df = df[df["patient"].isin(keep)]
            if df.empty:
                raise ValueError("No rows left after applying patient_filter.")

        self.df = df.reset_index(drop=True)
        self.transforms = image_transforms
        self.return_floats = return_floats
        self.dtype = dtype
        self.return_patient = return_patient

        # Determine gene columns explicitly (stable order)
        if genes is None:
            skip = {"tile", "x", "y", "patient"}
            candidates = []
            for c in self.df.columns:
                if c in skip or str(c).endswith("_lds_w"):
                    continue
                dt = getattr(self.df[c], "dtype", None)
                if dt is not None and getattr(dt, "kind", None) in {"f", "i", "u"}:
                    candidates.append(c)
            if not candidates:
                raise ValueError("Could not infer gene columns; please pass `genes`.")
            self.genes = list(candidates)
        else:
            missing = [g for g in genes if g not in self.df.columns]
            if missing:
                raise ValueError(f"Genes missing in DataFrame: {missing}")
            self.genes = list(genes)

    # add result to internal df, so that once everything is calculated one can plot the results spatially
    def add_result_for_tile(
        self,
        tile: str,
        result,
        *,
        column: str = None,
        prefix: str = None,
        create_columns: bool = True,
        use_first_if_duplicate: bool = True,
    ) -> int:
        # locate row(s)
        matches = self.df.index[self.df["tile"] == tile].tolist()
        if not matches:
            raise KeyError(f"No row found for tile={tile!r}.")
        if len(matches) > 1 and not use_first_if_duplicate:
            raise ValueError(f"Multiple rows found for tile={tile!r}: {matches}")
        idx = matches[0]

        def _ensure_col(colname: str):
            if colname not in self.df.columns:
                raise KeyError(f"Column {colname!r} does not exist and create_columns=False.")

        # Write logic
        if isinstance(result, Mapping):
            for k, v in result.items():
                _ensure_col(k)
                self.df.at[idx, k] = v
        elif isinstance(result, (Sequence, np.ndarray)) and not isinstance(result, (str, bytes)):
            if prefix is None:
                raise ValueError("For sequence/array results, please provide a `prefix`.")
            # create the necessary columns
            n = len(result)
            for i in range(n):
                _ensure_col(f"{prefix}{i}")
            for i, v in enumerate(result):
                self.df.at[idx, f"{prefix}{i}"] = v
        else:
            # scalar
            if column is None:
                raise ValueError("For scalar result, please provide `column`.")
            _ensure_col(column)
            self.df.at[idx, column] = result

        return idx

    # Batch version
    def add_results(self, tile_to_result: Mapping[str, object], **kwargs) -> list[int]:
        updated = []
        for t, r in tile_to_result.items():
            updated.append(self.add_result_for_tile(t, r, **kwargs))
        return updated

    # Convenience
    @property
    def patients(self) -> List[str]:
        return sorted(self.df["patient"].unique().tolist())

    def __len__(self):
        return len(self.df)

    def _load_image(self, path: str):
        img = Image.open(path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img

    def _row_to_target(self, row):
        vals = [float(row[g]) for g in self.genes]
        if self.return_floats:
            return np.asarray(vals, dtype=np.float32)
        return torch.tensor(vals, dtype=self.dtype)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        img = self._load_image(row["tile"])
        target = self._row_to_target(row)

        x_val = float(row["x"])
        y_val = float(row["y"])
        patient_id = row["patient"]

        if self.return_floats:
            if self.return_patient:
                return img, target, x_val, y_val, patient_id
            return img, target, x_val, y_val

        x_t = torch.tensor(x_val, dtype=self.dtype)
        y_t = torch.tensor(y_val, dtype=self.dtype)

        if self.return_patient:
            # keep patient as a Python str to avoid encoding assumptions
            return img, target, x_t, y_t, patient_id

        return img, target, x_t, y_t


def get_spatial_dataset(
    data_dir: str | Path,
    genes: list[str],
    *,
    transforms=T.ToTensor(),
    samples: list[str] | None = None,
    meta_data_dir: str = "/meta_data/",
    gene_data_filename: str = "gene_data.csv",
    return_floats: bool = False,
    max_len: int | None = None,
    patient_filter: str | list[str] | None = None,
    return_patient: bool = False,
):

    if samples is None:
        samples = [f.name for f in os.scandir(data_dir) if f.is_dir()]

    dfs = []
    for sample in samples:
        # load gene data
        g_path = os.path.join(data_dir, sample, meta_data_dir.lstrip("/"), gene_data_filename)
        g_df = pd.read_csv(g_path, nrows=max_len)

        g_df["patient"] = sample

        # load spatial coords
        s_path = os.path.join(data_dir, sample, meta_data_dir.lstrip("/"), "spatial_data.csv")
        s_df = pd.read_csv(s_path, usecols=["tile", "x", "y"])

        # merge on tile
        df = pd.merge(g_df, s_df, on="tile")
        df["tile"] = df["tile"].apply(lambda t: os.path.join(data_dir, sample, "tiles", t))
        dfs.append(df)

    base_df = pd.concat(dfs, ignore_index=True)

    ds = STSpatialDataset(
        base_df,
        image_transforms=transforms,
        genes=genes,
        return_floats=return_floats,
        patient_filter=patient_filter,
        return_patient=return_patient,
    )
    return ds
