import numpy as np
from lds_helpers import LDS        # assumes utils.py is on PYTHONPATH or same folder
from data_loader import label_dataset
import matplotlib.pyplot as plt
from scipy.stats import entropy

# may return None
def _get_smoothing(gene, dataset, method):
    if method not in {"gaussian", "triang", "laplace", "silverman"}:
        raise ValueError("smoothing method not implemented")

    try:
        labels = np.asarray(dataset[gene], dtype=float).ravel()
    except Exception as err:
        raise KeyError(f"Gene '{gene}' not found in dataset") from err

    bins_space = [b for b in range(5, 35)]
    ks_space   = [k for k in range(3, 11, 2)]

    best_js = -np.inf
    best_set = None

    kind = "gaussian" if method == "silverman" else method
    sigma = None if method in {"gaussian", "silverman"} else 2.0
    for bins_i in bins_space:
        for ks in ks_space:
            lds = LDS(bins=bins_i, ks=ks, kind=kind, sigma=sigma)
            js = lds.compare(labels, metric="js")

            if np.isfinite(js) and js > best_js:
                best_js = js
                best_hist, best_edges = lds(labels)
                best_set = (best_hist, best_edges, ks, sigma, bins_i, gene, method)

    # no default return because that is not that helpful and would be hard to distinguish
    # therefore can return None

    return best_set

def get_smoothings(genes, dataset, method):
    results = []
    for gene in genes:
        result = _get_smoothing(gene, dataset, method)
        assert(result is not None)
        results.append(result)
    return results

def prepare_weights(cfg):
    filenames = cfg["gene_data_filename"]
    for filename in filenames:
        dataset = label_dataset(cfg["datadir"], cfg["genes"][0], cfg["train_samples"], filename)
        results = get_smoothings(cfg["genes"], dataset, cfg["method"])
        for result in results:
            if result is not None:
                best_hist, best_edges, ks, sigma, bins_i, gene, method = result



                plt.hist()
                plt.title("ks: " + str(ks) + " sigma: " + str(sigma))
                plt.stairs(eff_label_dist, edges=bin_edges)
                plt.show()



from typing import Iterable, Tuple, Optional

import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal.windows import triang
from scipy.stats import entropy, wasserstein_distance


class LDS:
    EPS = 1e-12
    def __init__(
        self,
        binspace: int = 30,
        kspace: int = 5,
        kind: str = "gaussian",
        sigma: Optional[float] = 2.0,
    ) -> None:
        if ks % 2 == 0:
            raise ValueError("`ks` must be odd.")
        if kind not in {"gaussian", "triang", "laplace"}:
            raise ValueError("Unsupported kernel type.")
        self.bins, self.ks, self.kind, self.sigma = bins, ks, kind, sigma

    # ---------------------------------------------------------------------
    # internals
    # ---------------------------------------------------------------------
    def _kernel(self, sigma: float) -> np.ndarray:
        half = (self.ks - 1) // 2
        if self.kind == "gaussian":
            base = np.zeros(self.ks, dtype=np.float32)
            base[half] = 1.0  # impulse
            k = gaussian_filter1d(base, sigma, mode="constant")
        elif self.kind == "triang":
            k = triang(self.ks).astype(np.float32)
        else:  # laplace
            x = np.arange(-half, half + 1, dtype=np.float32)
            k = np.exp(-np.abs(x) / sigma)

        k /= k.max()  # safe: k.max()>0
        return k

    @staticmethod
    def _silverman_sigma(labels: np.ndarray) -> float:
        n = labels.size
        std = labels.std(ddof=1)
        iqr = np.subtract(*np.percentile(labels, [75, 25]))
        return 0.9 * min(std, iqr / 1.34) * n ** (-1 / 5)

    def _smooth(self, hist: np.ndarray, sigma: float) -> np.ndarray:
        return convolve1d(hist.astype(np.float32), self._kernel(sigma), mode="constant")

    def __call__(self, labels: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
        labels = np.asarray(labels, dtype=float).ravel()
        hist, edges = np.histogram(labels, bins=self.bins)

        if self.ks <= 1:
            return hist.astype(np.float32), edges

        sigma = (
            self.sigma
            if self.sigma is not None
            else (self._silverman_sigma(labels) if self.kind == "gaussian" else 1.0)
        )
        return self._smooth(hist, sigma), edges

    smooth = __call__  # convenience alias

    def compare(self, labels: Iterable[float], metric: str, show_plot: bool = True) -> float:
        labels = np.asarray(labels, dtype=float).ravel()
        empirical, edges = np.histogram(labels, bins=self.bins)
        smoothed, _ = self(labels)

        if empirical.sum() == 0 or smoothed.sum() == 0:
            return float("nan")

        p = np.maximum(empirical / empirical.sum(), self.EPS)
        q = np.maximum(smoothed / smoothed.sum(), self.EPS)
        p /= p.sum()
        q /= q.sum()

        if show_plot:
            import matplotlib.pyplot as plt

            centers = 0.5 * (edges[:-1] + edges[1:])
            fig, ax = plt.subplots()

            ax.step(centers, p, where="mid", label="empirical", lw=1.5)
            ax.step(centers, q, where="mid", label="smoothed",  lw=1.5)
            ax.set_xlabel("bin centre")
            ax.set_ylabel("probability")
            ax.set_title(f"LDS smoothing (bins={self.bins}, ks={self.ks})")
            ax.legend()
            if ax is None:  # created a fresh fig -> show it
                plt.show()

        metric = metric.lower()
        if metric == "js":  # Jensen–Shannon
            m = 0.5 * (p + q)
            return 0.5 * (entropy(p, m) + entropy(q, m))
        if metric == "kl":  # Kullback–Leibler
            return entropy(p, q)
        if metric == "wass":  # 1-Wasserstein / EMD
            centers = 0.5 * (edges[:-1] + edges[1:])
            return wasserstein_distance(
                centers.tolist(), centers.tolist(), p.tolist(), q.tolist()
            )
        raise ValueError("metric must be 'js', 'kl', or 'wass'")

