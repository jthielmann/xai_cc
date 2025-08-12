from __future__ import annotations

import os
import random
import numpy as np
import torch
import torch.nn.functional
from torchvision import transforms
from PIL import Image
from script.data_processing.image_transforms import get_transforms
from typing import List, Literal
from pathlib import Path
from pathlib import Path
from typing import Dict, List, Literal, Optional
import json
import pandas as pd
import torch
DEFAULT_RANDOM_SEED = 42

import numpy as np
from typing import Tuple, Optional, Sequence
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal.windows import triang
from scipy.stats import entropy, wasserstein_distance
from script.main_utils import parse_yaml_config, parse_args
from script.configs.dataset_config import get_dataset_data_dir

import pandas as pd
import os
from pathlib import Path
import torch


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


seed_everything(seed=DEFAULT_RANDOM_SEED)


def log_training(date, training_log):
    with open(training_log, "a") as f:
        f.write(
            date + " Resnet50 - single gene\top: AdamW\telrs: 0.9\tlfn: MSE Loss\n")  # Adapt to model and gene name(s) getting trained


import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Optional

class STDataset(Dataset):
    def __init__(
        self,
        df,
        *,
        image_transforms=None,
        inputs_only: bool = False,
        genes: Optional[List[str]] = None,
        use_weights: bool = False,              # if True, return (img, y, w)
    ):
        self.df = df.reset_index(drop=True)
        self.transforms = image_transforms
        self.inputs_only = inputs_only

        # Determine gene columns explicitly (stable order)
        if genes is None:
            # auto-detect genes = all numeric columns except 'tile' and *_lds_w
            candidates = [c for c in self.df.columns if c != "tile" and not str(c).endswith("_lds_w")]
            if not candidates:
                raise ValueError("Could not infer gene columns; please pass `genes`.")
            self.genes = list(candidates)
        else:
            missing = [g for g in genes if g not in self.df.columns]
            if missing:
                raise ValueError(f"Genes missing in DataFrame: {missing}")
            self.genes = list(genes)

        self.G = len(self.genes)

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
            # Return image only; caller can ignore targets entirely
            return img

        y = self._row_to_target(row)             # shape (G,)

        if self.use_weights:
            w = self._row_to_weights(row)        # shape (G,)
            return img, y, w

        return img, y


class STDataset_umap(STDataset):
    def __init__(self, dataframe, image_transforms=None, device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu", inputs_only=False):
        super().__init__(dataframe, image_transforms, device, inputs_only)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        a = Image.open(row["tile"]).convert("RGB")
        if self.transforms:
            a = self.transforms(a)
        if self.device_handling:
            a = a.to(self.device)
        return a, row["tile"]


class TileLoader:
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # mean and std of the whole dataset
            transforms.Normalize([0.7406, 0.5331, 0.7059], [0.1651, 0.2174, 0.1574])
            ])

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
from typing import Dict

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
        w2 = (w2 / w2.mean()).clamp(max=clip_max)
        scaled[g] = w2.to(dtype=torch.float32)
    return scaled



def get_all_genes():
    pass

# contains tile path and gene data
def get_base_dataset(
    data_dir: str | Path,
    genes: List[str],
    samples: List[str],
    *,
    meta_data_dir: str = "/meta_data/",
    max_len: int | None = None,
    bins: int = 1,
    gene_data_filename: str = "gene_data.csv",
    lds_smoothing_csv: str | Path | None = None,      # <─ NOW one file, not a dir
    weight_transform: str = "inverse",
    weight_clamp: int = 10
):
    columns_of_interest = ["tile"] + (genes or []) if genes else None
    dfs = []

    for patient in samples:
        fp = os.path.join(data_dir, patient, meta_data_dir.lstrip("/"), gene_data_filename)
        df = pd.read_csv(fp, usecols=columns_of_interest, nrows=max_len)
        df["tile"] = df["tile"].apply(lambda t: os.path.join(data_dir, patient, "tiles", t))
        df["patient"] = patient
        dfs.append(df)

    base_df = pd.concat(dfs, ignore_index=True)

    if lds_smoothing_csv is not None:
        gene2weights = load_gene_weights(lds_smoothing_csv, genes=genes,weight_transform=weight_transform)
        gene2weights = make_weights(gene2weights, clip_max=weight_clamp, r1=20, r2=100)

        for g in genes:
            if g not in gene2weights: raise ValueError(f"No weights for {g}")
            w_vec  = gene2weights[g]                      # (K,)
            K      = len(w_vec)

            vals   = base_df[g].to_numpy()
            # edges exactly like np.histogram (equal width)
            edges  = np.linspace(vals.min(), vals.max(), K + 1, dtype=np.float32)
            idx    = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, K - 1)
            base_df[f"{g}_lds_w"] = w_vec[idx]

    if bins > 1:
        gene_values = base_df[genes[0]]
        gene_bins   = pd.cut(gene_values, bins)
        groups      = base_df.groupby(gene_bins)
        sample_size = groups.size().max()
        base_df     = (groups
                       .apply(lambda x: x.sample(sample_size, replace=True) if len(x) else x)
                       .reset_index(drop=True))

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
    genes: list[str],
    *,
    transforms=None,
    samples: list[str] | None = None,
    meta_data_dir: str = "/meta_data/",
    max_len: int | None = None,
    bins: int = 1,
    only_inputs: bool = False,
    gene_data_filename: str = "gene_data.csv",
    lds_smoothing_csv: str | Path | None = None,
    weight_transform: str = "inverse",
    weight_clamp: int = 10
):
    if samples is None:
        samples = [f.name for f in os.scandir(data_dir) if f.is_dir()]

    df = get_base_dataset(
        data_dir,
        genes,
        samples,
        meta_data_dir=meta_data_dir,
        max_len=max_len,
        bins=bins,
        gene_data_filename=gene_data_filename,
        lds_smoothing_csv=lds_smoothing_csv,
        weight_transform=weight_transform,
        weight_clamp=weight_clamp
    )

    # create the PyTorch-compatible dataset
    ds = STDataset(
        df,
        image_transforms=transforms,
        inputs_only=only_inputs,
        genes=genes,
        use_weights=lds_smoothing_csv is not None,
    )
    return ds


# returns imgs and tile paths
def get_dataset_for_umap(data_dir, genes, transforms=None, samples=None, meta_data_dir="/meta_data/", max_len=None, bins=1, only_inputs=False):
    if samples is None:
        samples = [os.path.basename(f) for f in os.scandir(data_dir) if f.is_dir()]
    gene_data_df = get_base_dataset(data_dir, genes, samples, meta_data_dir=meta_data_dir, max_len=max_len, bins=bins)
    st_dataset = STDataset_umap(gene_data_df, image_transforms=transforms, inputs_only=only_inputs)
    return st_dataset


def get_dino_dataset(csv_path, dino_transforms=None, max_len=None, bins=1, device_handling=False):
    if dino_transforms is None:
        dino_transforms = get_transforms()
    file_df = pd.read_csv(csv_path, nrows=max_len)
    st_dataset = STDataset(file_df, image_transforms=dino_transforms, inputs_only=True, device_handling=device_handling)
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

# returns imgs and tile paths
def get_dataset_NCT_CRC_classification(data_dir, transforms=None, samples=None, meta_data_dir="/meta_data/", max_len=None):
    if samples is None:
        samples = [os.path.basename(f) for f in os.scandir(data_dir) if f.is_dir()]
    st_dataset = STDataset_umap(gene_data_df, image_transforms=transforms)
    return st_dataset






def get_label_dataframe(data_dir, samples, meta_data_dir="/meta_data/", max_len=None, gene_data_filename="gene_data.csv"):
    datasets = []  # Use a list to store DataFrames

    for i in samples:
        file_path = data_dir + i + meta_data_dir + gene_data_filename
        # Read the CSV file, excluding the 'tile' column because it is not needed for label smoothing
        st_dataset_patient = pd.read_csv(file_path, nrows=max_len, usecols=lambda col: col != "tile")
        datasets.append(st_dataset_patient)
    st_dataset = pd.concat(datasets)
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










class LDS:
    EPS: float = 1e-12

    def __init__(self, kernel_type: str, dataset: label_dataset, main_metric) -> None:
        self.kernel_type = kernel_type  # for _kernel_size
        self.best = None
        self.dataset = dataset
        self.main_metric = main_metric

    def _get_kernel(self, kernel_size: int, sigma: float) -> np.ndarray:
        """
        Return a 1‑D smoothing kernel of length `kernel_size`.
        `sigma` must already be expressed in **bin indices**, not in data units.
        The kernel is L1‑normalised (∑k = 1) so it preserves total mass.
        """
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if sigma <= 0:
            raise ValueError("sigma must be > 0.")

        half = (kernel_size - 1) // 2

        if self.kernel_type in {"gaussian", "silverman"}:
            # centred Gaussian: exp(-½ (x/σ)²)
            x = np.arange(-half, half + 1, dtype=np.float32)
            k = np.exp(-0.5 * (x / sigma) ** 2)

        elif self.kernel_type == "triang":
            # symmetric triangular window
            k = triang(kernel_size).astype(np.float32)

        elif self.kernel_type == "laplace":
            # Laplace / double‑exponential: exp(-|x|/σ)
            x = np.arange(-half, half + 1, dtype=np.float32)
            k = np.exp(-np.abs(x) / sigma)

        else:
            raise ValueError(f"Unsupported kernel type '{self.kernel_type}'")

        k /= k.sum()  # L1 normalisation
        return k

    def _get_silverman_sigma(self) -> float:
        labels = np.stack([self.dataset[i] for i in range(len(self.dataset))])
        n = labels.size
        std = labels.std(ddof=1)
        iqr = np.subtract(*np.percentile(labels, [75, 25]))
        return 0.9 * min(std, iqr / 1.34) * n ** (-1 / 5)

    def _smooth(self, hist: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
        min_len = int(np.ceil(6 * sigma)) | 1  # make it odd
        ks = max(kernel_size, min_len)
        kernel = self._get_kernel(ks, sigma)
        return convolve1d(hist.astype(np.float32), kernel, mode="reflect")


    def set_gene(self, gene):
        self.dataset.set_gene(gene)

    def _bandwidth_to_bins(self, bw_data_units: float, bins: int, data: np.ndarray) -> float:
        # convert “real” bandwidth to index bandwidth
        bin_width = (data.max() - data.min()) / bins
        return max(bw_data_units / bin_width, 1e-6)  # keep σ > 0

    def get_smoothing(self, bins: int, kernel_size: int, sigma:any=None) -> np.ndarray:
        if kernel_size <= 1:
            raise ValueError("Kernel size must be greater than 1.")

        labels = np.asarray([self.dataset[i] for i in range(len(self.dataset))], dtype=float)
        emp_hist, _ = np.histogram(labels, bins=bins)

        if self.kernel_type == "silverman":
            bw = self._get_silverman_sigma()  # data units
            sigma_idx = self._bandwidth_to_bins(bw, bins, labels)
        else:  # gaussian | triang | laplace
            # self.sigma is already meant to be in data units
            sigma_idx = self._bandwidth_to_bins(sigma, bins, labels)

        return self._smooth(emp_hist, kernel_size, sigma_idx)

    def get_empirical_distr(self, bins):
        labels = np.stack([self.dataset[i] for i in range(len(self.dataset))])
        emp_hist, emp_edges = np.histogram(labels, bins=bins)
        return emp_hist, emp_edges

    def rate_smoothing(self, smoothed_distr, empirical_distr, empirical_edges):
        if empirical_distr.sum() == 0 or smoothed_distr.sum() == 0:
            return np.nan, np.nan, np.nan, np.nan

        p = np.maximum(empirical_distr / empirical_distr.sum(), self.EPS)
        q = np.maximum(smoothed_distr / smoothed_distr.sum(), self.EPS)
        p /= p.sum()
        q /= q.sum()

        m = 0.5 * (p + q)
        js = 0.5 * (entropy(p, m) + entropy(q, m))
        kl = entropy(p, q)
        centres = 0.5 * (empirical_edges[:-1] + empirical_edges[1:])
        ws = wasserstein_distance(centres.tolist(), centres.tolist(), p, q)
        return js, kl, centres, ws

    @staticmethod
    def _load_cache(csv_path: Path) -> pd.DataFrame:
        """Return an empty DataFrame with the correct columns if the file is absent."""
        if csv_path.is_file():
            return pd.read_csv(csv_path)
        return pd.DataFrame(columns=["gene", "bins", "kernel_size", "sigma", "loo_ll"])

    @staticmethod
    def _append_cache(csv_path: Path, row: dict) -> None:
        """Append *row* to *csv_path*, creating the file if necessary."""
        df = LDS._load_cache(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(csv_path, index=False)

    # --------------- plotting utils (unchanged) ---------------
    def compareTo(self, gene, bins, kernel_size, sigma, ll_rating, rating) -> float:

        empirical, edges = self.get_empirical_distr(bins)

        smoothed = self._smooth(empirical.copy(), kernel_size, sigma)

        if empirical.sum() == 0 or smoothed.sum() == 0:
            return float("nan")

        p = np.maximum(empirical / empirical.sum(), self.EPS)
        q = np.maximum(smoothed / smoothed.sum(), self.EPS)
        p /= p.sum(); q /= q.sum()

        js, kl, centres, ws = rating

        fig, ax = plt.subplots()
        ax.step(centres, p, where="mid", lw=1.5, label="empirical")
        ax.step(centres, q, where="mid", lw=1.5, label="smoothed")
        ax.set_xlabel("bin centre")
        ax.set_ylabel("probability")
        ax.set_title(
            f"LDS (bins={bins}, ks={kernel_size}, sigma={sigma:.2g}, gene={gene}, ll={ll_rating:.2g})"
        )

        metrics_txt = f"JS  = {js:.4g}\nKL  = {kl:.4g}\nWass = {ws:.4g}"
        ax.annotate(
            metrics_txt,
            xy=(0.01, 0.99),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.7),
        )
        ax.legend()
        plt.show()

    def get_ll_rating(self, bins, ks, sigma):
        labels = np.asarray([self.dataset[i] for i in range(len(self.dataset))], dtype=float)
        full_hist, edges = np.histogram(labels, bins=bins)
        bin_idx = np.clip(
            np.searchsorted(edges, labels, side="right") - 1,
            0, bins - 1
        )

        # -----------------------------------------------------------------
        # 2. LOO loop: remove one point from its bin, smooth, evaluate p(x)
        # -----------------------------------------------------------------
        ll   = 0.0
        hist = full_hist.copy()                   # single reusable buffer

        for idx in bin_idx:
            hist[idx] -= 1                        # leave one out
            smooth    = self._smooth(hist, ks, sigma)

            # probability mass assigned to the left-out sample
            p_x = max(smooth[idx] / smooth.sum(), self.EPS)
            ll += np.log(p_x)

            hist[idx] += 1                        # restore for next sample

        return ll
    # -----------------------------------------------------------


# --------------------------- main -----------------------------
if __name__ == "__main__":
    args = parse_args()
    bin_space = [20, 30, 40, 50, 100]        # number of histogram bins to test
    ks_space = [7, 9, 11]          # odd kernel sizes
    sigma_space = [0.8, 1.0, 1.2, 1.5, 2.0] # search space for σ (in index units)

    cfg = parse_yaml_config(args.config)
    params = cfg.get("parameters", {})

    data_dir           = get_dataset_data_dir(params["dataset"].get("value"))
    gene_data_filename = params["gene_data_filename"].get("value")
    max_len            = 100 if params["debug"].get("value") else None
    samples            = params["train_samples"].get("value")
    debug              = params["debug"].get("value")
    if debug:
        samples = samples[:1]

    dataset = label_dataset(
        data_dir=data_dir,
        samples=samples,
        gene_data_filename=gene_data_filename,
        max_len=max_len,
    )

    lds = LDS(
        kernel_type="gaussian",
        dataset=dataset,
        main_metric="js",
    )

    genes = [
        "TNNT1", "AQP5", "RAMP1", "ADGRG6", "SECTM1", "DPEP1", "CHP2", "RUBCNL",
        "SLC9A3", "VAV3", "MUC2", "PIGR", "TFF1", "KIAA1324", "ZBTB7C", "SERPINA1",
        "SPOCK1", "FBLN1", "ANTXR1", "TNS1", "MYL9", "HSPB8",
    ]

    # ---------- NEW: prepare cache ----------
    cache_path = Path("best_smoothings.csv")

    if cache_path.is_file():
        cache_df = pd.read_csv(cache_path)
    else:
        cache_df =  pd.DataFrame(columns=["gene", "bins", "kernel_size", "sigma", "loo_ll"])

    for gene in genes:
        lds.set_gene(gene)
        for bins in bin_space:
            for ks in ks_space:
                for sigma in sigma_space:
                    if not cache_df[
                        (cache_df["gene"] == gene) &
                        (cache_df["bins"] == bins) &
                        (cache_df["kernel_size"] == ks) &
                        (cache_df["sigma"] == sigma)
                        ].empty:
                        continue

                    lds.set_gene(gene)
                    emp, edges = lds.get_empirical_distr(bins)
                    smoothed = lds.get_smoothing(bins, ks, sigma)
                    rating = lds.rate_smoothing(smoothed, emp, edges)
                    ll_rating = lds.get_ll_rating(bins, ks, sigma)

                    cache_df.loc[len(cache_df)] = {
                        "gene": gene,
                        "bins": bins,
                        "kernel_size": ks,
                        "sigma": sigma,
                        "loo_ll": ll_rating
                    }
                    cache_df.to_csv(cache_path, index=False)

    best_per_gene = cache_df.loc[
        cache_df.groupby("gene")["loo_ll"].idxmax()
    ].reset_index(drop=True)

    print("===  Best hyper-parameters per gene  ===")
    print(best_per_gene)  # or log to a file if you prefer

    for _, row in best_per_gene.iterrows():
        gene = row["gene"]
        bins = int(row["bins"])
        ks = int(row["kernel_size"])
        sigma = float(row["sigma"])
        ll_rating = float(row["loo_ll"])

        # make sure LDS is focused on the right gene
        lds.set_gene(gene)
        emp, edges = lds.get_empirical_distr(bins)
        smoothed = lds.get_smoothing(bins, ks, sigma)
        rating = lds.rate_smoothing(smoothed, emp, edges)

        # your custom comparison / evaluation
        lds.compareTo(gene, bins, ks, sigma, ll_rating, rating)


from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, Literal

def load_best_smoothing(csv_path: str | Path, gene: str) -> Dict[str, Any]:

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    # keep only the rows for this gene
    sub = df[df["gene"] == gene]
    if sub.empty:
        raise KeyError(f"Gene {gene!r} not present in {csv_path}")

    # pick the row with the highest leave-one-out log-likelihood
    row = sub.loc[sub["loo_ll"].idxmax()]

    weights = np.array(json.loads(row["weights_json"]), dtype=float)
    best = {
        "bins": int(row["bins"]),
        "kernel_size": int(row["kernel_size"]),
        "sigma": float(row["sigma"]),
        "loo_ll": float(row["loo_ll"]),
        "weights": weights,
    }
    return best


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
) -> Dict[str, torch.Tensor]:
    """
    Build {gene: weight_tensor} from the LDS CSV.

    Parameters
    ----------
    csv_path : path to CSV with columns ["gene", "weights_json", selection_metric, ...]
    genes : list of genes to load
    weight_transform :
        - "inverse"       -> w = 1 / p
        - "sqrt-inverse"  -> w = 1 / sqrt(p)
        - "none"          -> w = p
      (p is the smoothed distribution; we always re-normalize p to sum=1 first.)
    selection_metric : column name to choose the best row per gene (default: "js")
    mode : "min" or "max" depending on the metric direction (JS uses "min")
    eps : numerical floor for stability
    renorm_mean1 : if True, scale weights so mean(w) == 1 (keeps loss scale stable)
    clip_max : optional upper bound to clip extreme weights

    Returns
    -------
    Dict[str, torch.Tensor] mapping gene -> 1D float32 tensor of length == bins.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

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

    return gene2weights


class PlottingDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, device,
                 transforms=transforms.Compose([
                     transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     # mean and std of the whole dataset
                     transforms.Normalize([0.7406, 0.5331, 0.7059],
                                          [0.1651, 0.2174, 0.1574])
                 ])):
        self.dataframe = dataframe
        self.transforms = transforms
        self.device = device

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        gene_names = list(self.dataframe)[1:]
        gene_vals = []
        row = self.dataframe.iloc[index]
        a = Image.open(row["tile"]).convert("RGB")
        for j in gene_names:
            gene_val = float(row[j])
            gene_vals.append(gene_val)
        e = row["tile"]
        # apply normalization transforms as for pretrained colon classifier
        a = self.transforms(a)
        a = a.to(self.device)
        return a, 0


# has the labels set to 0 because that makes it easier to work with the frameworks written for classification
# the idea is that they filter the attribution by the chosen class, but as we only have one output we always choose y=0
def get_dataset_for_plotting(data_dir, genes, samples=None,
                             device="cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu"):

    if samples is None:
        samples = []
        for subdir in os.listdir(data_dir):
            if os.path.isdir(data_dir + "/" + subdir):
                samples.append(subdir)

    columns_of_interest = ["tile"]
    for gene in genes:
        columns_of_interest.append(gene)
    dataset = pd.DataFrame(columns=columns_of_interest)

    # generate training dataframe with all training samples
    for i in samples:
        st_dataset = pd.read_csv(data_dir + "/" + i + "/meta_data/gene_data.csv", index_col=-1)
        st_dataset["tile"] = st_dataset.index
        st_dataset["tile"] = st_dataset["tile"].apply(
            lambda x: str(data_dir) + "/" + str(i) + "/tiles/" + str(x)
        )
        if dataset.empty:
            dataset = st_dataset[columns_of_interest]
        else:
            # concat
            dataset = pd.concat([dataset, st_dataset[columns_of_interest]])

    # reset index of dataframes
    dataset.reset_index(drop=True, inplace=True)

    return PlottingDataset(dataset, device=device)