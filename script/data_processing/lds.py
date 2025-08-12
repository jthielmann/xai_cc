from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import convolve1d
from scipy.stats import entropy, iqr

from script.data_processing.data_loader import label_dataset
from script.main_utils import parse_yaml_config, parse_args
from script.configs.dataset_config import get_dataset_data_dir, get_dataset_cfg_lds


@dataclass(frozen=True)
class LDSParams:
    bins: int
    kernel_size: int
    sigma: float


class LDS:
    """Label Distribution Smoothing utility with cached labels/histograms."""
    EPS = 1e-12

    def __init__(self, kernel_type: str, dataset):
        """
        dataset must support:
          - __len__()
          - __getitem__(i) -> label (float)
          - set_gene(gene: str)
        """
        self.kernel_type = kernel_type
        self.dataset = dataset
        self._label_cache: Dict[str, np.ndarray] = {}
        self._hist_cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}

    # ---------- internals ----------
    @staticmethod
    def _ensure_odd(n: int) -> int:
        n = max(int(n), 3)
        return n if (n % 2 == 1) else n + 1

    def _labels_for_gene(self, gene: str) -> np.ndarray:
        if gene not in self._label_cache:
            self.dataset.set_gene(gene)
            # pulling labels once per gene is much faster than per-try
            labels = np.asarray([self.dataset[i] for i in range(len(self.dataset))],
                                dtype=np.float32)
            self._label_cache[gene] = labels
        return self._label_cache[gene]

    def _empirical(self, gene: str, bins: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (gene, bins)
        if key not in self._hist_cache:
            labels = self._labels_for_gene(gene)
            self._hist_cache[key] = np.histogram(labels, bins=bins)
        return self._hist_cache[key]

    def _kernel(self, kernel_size: int, sigma_idx: float) -> np.ndarray:
        ks = self._ensure_odd(kernel_size)

        if self.kernel_type in {"gaussian", "silverman"}:
            half = (ks - 1) // 2
            x = np.arange(-half, half + 1, dtype=np.float32)
            k = np.exp(-0.5 * (x / max(sigma_idx, self.EPS)) ** 2)
        elif self.kernel_type == "triang":
            from scipy.signal.windows import triang
            k = triang(ks).astype(np.float32)
        elif self.kernel_type == "laplace":
            half = (ks - 1) // 2
            x = np.arange(-half, half + 1, dtype=np.float32)
            k = np.exp(-np.abs(x) / max(sigma_idx, self.EPS))
        else:
            raise ValueError(f"Unsupported kernel type '{self.kernel_type}'")

        s = k.sum()
        return k if s == 0 else (k / s)

    def _sigma_to_bin_units(self, gene: str, bins: int, sigma: float) -> float:
        """If kernel_type == 'silverman', convert continuous bandwidth to bin units."""
        if self.kernel_type != "silverman":
            return float(sigma)

        labels = self._labels_for_gene(gene)
        if labels.size < 2:
            return 1.0  # fallback

        std = float(labels.std(ddof=1))
        spread = min(std, float(iqr(labels)) / 1.34)
        bw = 0.9 * spread * (labels.size ** (-1 / 5))  # Silverman’s rule of thumb
        # Convert to bin indices
        lo, hi = float(labels.min()), float(labels.max())
        bin_width = max((hi - lo) / bins, self.EPS)
        return max(bw / bin_width, 1e-3)

    # ---------- public API ----------
    def get_smoothing(self, gene: str, params: LDSParams) -> np.ndarray:
        emp_hist, _ = self._empirical(gene, params.bins)
        sigma_idx = self._sigma_to_bin_units(gene, params.bins, params.sigma)
        kernel = self._kernel(params.kernel_size, sigma_idx)
        smoothed = convolve1d(emp_hist.astype(np.float32), kernel, mode="reflect")
        return smoothed

    @staticmethod
    def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        p = np.maximum(p.astype(np.float64), 0)
        q = np.maximum(q.astype(np.float64), 0)
        ps = p.sum()
        qs = q.sum()
        if ps <= LDS.EPS or qs <= LDS.EPS:
            return 0.0  # degenerate but defined
        p /= ps
        q /= qs
        m = 0.5 * (p + q)
        return 0.5 * (entropy(p, m) + entropy(q, m))

    def compare_plot(self, gene: str, params: LDSParams, weights: np.ndarray, out_path: Path):
        emp_hist, edges = self._empirical(gene, params.bins)
        centres = 0.5 * (edges[:-1] + edges[1:])
        p = emp_hist / max(emp_hist.sum(), self.EPS)
        q = weights / max(weights.sum(), self.EPS)

        plt.figure()
        plt.step(centres, p, where="mid", label="empirical")
        plt.step(centres, q, where="mid", label="smoothed")
        plt.xlabel("Value")
        plt.ylabel("Probability")
        plt.title(f"{gene} (bins={params.bins}, ks={params.kernel_size}, sigma={params.sigma})")
        plt.legend()
        out_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path / f"{gene}_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()


def grid_search_lds(
    lds: LDS,
    genes: Iterable[str],
    bin_space: Iterable[int],
    ks_space: Iterable[int],
    sigma_space: Iterable[float],
) -> pd.DataFrame:
    """Return best params per gene (by JS divergence) + weights as JSON."""
    records: List[Dict] = []

    for gene in genes:
        best_js = float("inf")
        best_row: Dict | None = None

        # cache labels/hists once per (gene,bins)
        _ = lds._labels_for_gene(gene)  # ensure cache
        for bins in bin_space:
            emp_hist, _ = lds._empirical(gene, bins)
            for ks, sigma in product(ks_space, sigma_space):
                params = LDSParams(bins=bins, kernel_size=ks, sigma=sigma)
                weights = lds.get_smoothing(gene, params)
                js = lds.js_divergence(emp_hist, weights)
                if js < best_js:
                    best_js = js
                    best_row = dict(
                        gene=gene,
                        bins=bins,
                        kernel_size=LDS._ensure_odd(ks),
                        sigma=float(sigma),
                        js=float(js),
                        weights=weights.astype(float).tolist(),
                    )

        assert best_row is not None, "No best row found (empty search space?)"
        records.append(best_row)

    df = pd.DataFrame.from_records(records)
    df["weights_json"] = df["weights"].apply(json.dumps)
    return df


def main():
    args = parse_args()
    cfg = parse_yaml_config(args.config)
    ds_cfg = get_dataset_cfg_lds(cfg.get("parameters", {}))
    cfg.get("parameters", {}).update(ds_cfg)
    params = cfg.get("parameters", {})

    data_dir = get_dataset_data_dir(params["dataset"].get("value"))
    dataset = label_dataset(
        data_dir=data_dir,
        samples=params["train_samples"],
        gene_data_filename=params["gene_data_filename"].get("values")[0],
        max_len=100 if params.get("debug", {}).get("value") else None,
    )

    # ---- config you likely tweak ----
    kernel_type = "gaussian"  # {"gaussian","silverman","triang","laplace"}
    genes = [
        "TNNT1", "AQP5", "RAMP1", "ADGRG6", "SECTM1", "DPEP1", "CHP2",
        "RUBCNL", "SLC9A3", "VAV3", "MUC2", "PIGR", "TFF1", "KIAA1324",
        "ZBTB7C", "SERPINA1", "SPOCK1", "FBLN1", "ANTXR1", "TNS1",
        "MYL9", "HSPB8"
    ]
    bin_space = [20, 30, 40, 50, 100]
    ks_space = [7, 9, 11]
    sigma_space = [0.8, 1.0, 1.2, 1.5, 2.0]

    out_csv = Path("best_smoothing.csv")
    out_plots = Path("lds_plots")

    lds = LDS(kernel_type=kernel_type, dataset=dataset)
    df = grid_search_lds(lds, genes, bin_space, ks_space, sigma_space)

    # keep only what you need in CSV
    df_out = df.drop(columns=["weights"]).copy()
    df_out.to_csv(out_csv, index=False)

    # generate plots (optional)
    for _, row in df.iterrows():
        params = LDSParams(
            bins=int(row["bins"]),
            kernel_size=int(row["kernel_size"]),
            sigma=float(row["sigma"]),
        )
        weights = np.array(json.loads(row["weights_json"]), dtype=np.float32)
        lds.compare_plot(str(row["gene"]), params, weights, out_path=out_plots)


if __name__ == "__main__":
    main()