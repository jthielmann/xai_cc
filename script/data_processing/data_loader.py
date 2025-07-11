import os
import random
import numpy as np
import torch
import torch.nn.functional
import pandas as pd
from torchvision import transforms
from PIL import Image
from script.data_processing.image_transforms import get_transforms
from typing import List, Literal
from pathlib import Path

DEFAULT_RANDOM_SEED = 42


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


class STDataset(torch.utils.data.Dataset):
    def __init__(self, df, *, image_transforms=None,
                 inputs_only=False, genes=None, use_weights=False):
        self.df   = df.reset_index(drop=True)
        self.genes = genes
        self.transforms = image_transforms
        self.inputs_only = inputs_only


        if use_weights and genes:
            w_col = f"{genes[0]}_lds_w"
            if w_col in self.df.columns:
                self.weights = torch.tensor(self.df[w_col].values,
                                            dtype=torch.float32)
            else:                               # weights asked but missing
                raise ValueError(
                    f"Column {w_col} not found – "
                    "did you forget lds_weight_dir?"
                )
        else:
            self.weights = torch.ones(len(self.df), dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def get_tilename(self, index):
        return self.df.iloc[index]["tile"]

    def __getitem__(self, index):
        cols = list(self.df)[:-1]
        gene_names = [c for c in cols if 'tile' not in c and '_lds_w' not in c]
        row = self.df.iloc[index]
        img = Image.open(row["tile"]).convert("RGB")
        # print(x.size)
        if self.transforms:
            img = self.transforms(img)
        # clustering uses crp which currently only supports classification tasks. therefore we take class 0 as a hack
        # as we only have one gene output in clustering because we cluster each gene separately
        if self.inputs_only:
            return img, 0
        gene_vals = []
        for j in gene_names:
            #gene_vals.append(float(row[j]))
            #mps:
            gene_val = torch.tensor(float(row[j]), dtype=torch.float32)
            #gene_val = torch.tensor(float(row[j]))
            gene_vals.append(gene_val)

        return img, torch.stack(gene_vals)


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
):
    """
    Build a single dataframe that optionally contains LDS-based weights.

    Returns
    -------
    pd.DataFrame
        Columns: tile, <gene>, …, <gene>_lds_w (if weights requested)
    """
    columns_of_interest = ["tile"] + (genes or [])
    dfs = []

    # 1) gather per-patient CSVs ------------------------------------------------
    for patient in samples:
        fp = os.path.join(data_dir, patient, meta_data_dir.lstrip("/"), gene_data_filename)
        df = pd.read_csv(fp, usecols=columns_of_interest, nrows=max_len)
        df["tile"] = df["tile"].apply(lambda t: os.path.join(data_dir, patient, "tiles", t))
        dfs.append(df)

    base_df = pd.concat(dfs, ignore_index=True)

    # 2) attach LDS weights -----------------------------------------------------
    if lds_smoothing_csv is not None:
        gene2weights = load_gene_weights(lds_smoothing_csv, genes=genes,weight_transform=weight_transform)

        for g in genes:
            w_vec  = gene2weights[g]                      # (K,)
            K      = len(w_vec)

            vals   = base_df[g].to_numpy()
            # edges exactly like np.histogram (equal width)
            edges  = np.linspace(vals.min(), vals.max(), K + 1, dtype=np.float32)
            idx    = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, K - 1)
            base_df[f"{g}_lds_w"] = w_vec[idx]

    # 3) optional equal-bin resampling -----------------------------------------
    if bins > 1:
        gene_values = base_df[genes[0]]
        gene_bins   = pd.cut(gene_values, bins)
        groups      = base_df.groupby(gene_bins)
        sample_size = groups.size().max()
        base_df     = (groups
                       .apply(lambda x: x.sample(sample_size, replace=True) if len(x) else x)
                       .reset_index(drop=True))

    return base_df



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
    lds_smoothing_csv: str | Path | None = None,   # ← one file, not a dir
    weight_transform: str = "inverse",             # ← "inverse" | "sqrt-inverse" | "none"
):
    """
    Wrapper that returns an STDataset whose dataframe optionally contains
    <gene>_lds_w columns built from best_smoothings.csv.
    """
    # discover patients if the caller did not supply them
    if samples is None:
        samples = [f.name for f in os.scandir(data_dir) if f.is_dir()]

    # build the dataframe (adds LDS weights if csv provided)
    df = get_base_dataset(
        data_dir,
        genes,
        samples,
        meta_data_dir=meta_data_dir,
        max_len=max_len,
        bins=bins,
        gene_data_filename=gene_data_filename,
        lds_smoothing_csv=lds_smoothing_csv,
        weight_transform=weight_transform
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







import numpy as np
from typing import Tuple, Optional, Sequence
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal.windows import triang
from scipy.stats import entropy, wasserstein_distance
from script.main_utils import parse_yaml_config, parse_args
from script.configs.dataset_config import get_dataset_data_dir

# NEW: lightweight caching of the best parameters
import pandas as pd
import os
from pathlib import Path
import torch


"""
    LDS –  Light Distribution Smoother
    ---------------------------------
    Added capability to cache the best smoothing parameters (bins, kernel_size, sigma_idx, loo_ll)
    for each gene in a simple CSV file. On the next run, the script will re‑use the cached values
    and skip the expensive leave‑one‑out search unless a gene is missing from the cache.
"""

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
    weight_transform: str = "inverse",
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Read the CSV produced by LDS and build {gene: weight_tensor}.

    Parameters
    ----------
    csv_path : str | Path
        CSV with columns …, weights_json.
    weight_transform : {"inverse", "sqrt-inverse", "none"}
        How to turn a probability vector p into weights w:
        * "inverse"      -> w = 1 / p
        * "sqrt-inverse" -> w = 1 / sqrt(p)
        * "none"         -> w = p  (use the smoothing itself as weights)
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    Dict[str, torch.Tensor]
        Each tensor has dtype=float32 and shape (K,), where K==bins
        for that gene, ready to feed into MultiGeneWeightedMSE.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    # take the row with the best LOO-LL for each gene
    best_rows = df.loc[df.groupby("gene")["loo_ll"].idxmax()]

    gene2weights: Dict[str, torch.Tensor] = {}

    for _, row in best_rows.iterrows():
        gene   = row["gene"]
        if gene not in genes:
            continue
        p      = torch.tensor(json.loads(row["weights_json"]), dtype=torch.float32)

        # normalise to a proper probability vector (∑1)
        p /= p.sum().clamp_min(eps)

        if weight_transform == "inverse":
            w = 1.0 / p.clamp_min(eps)
        elif weight_transform == "sqrt-inverse":
            w = 1.0 / p.clamp_min(eps).sqrt()
        elif weight_transform == "none":
            w = p.clone()
        else:
            raise ValueError(f"Unknown weight_transform: {weight_transform}")

        gene2weights[gene] = w

    return gene2weights

