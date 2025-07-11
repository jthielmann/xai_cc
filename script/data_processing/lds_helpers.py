from typing import Iterable, Tuple, Optional

import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal.windows import triang
from scipy.stats import entropy, wasserstein_distance


class LDS:
    EPS = 1e-12
    def __init__(
        self,
        bins: int = 30,
        ks: int = 5,
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
