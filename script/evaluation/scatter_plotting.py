import os
from typing import Optional

import numpy as np
from matplotlib import cm, colors, pyplot as plt
import pandas as pd


def make_scatter_figure(yh: np.ndarray, yt: np.ndarray, loss: float, r: float, title: str):
    lo, hi = float(min(yh.min(), yt.min())), float(max(yh.max(), yt.max()))
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    fig, ax = plt.subplots()
    coords = np.column_stack((yh, yt))
    unique_coords, counts = np.unique(coords, axis=0, return_counts=True)
    cmap = cm.get_cmap("Blues")
    vmax = max(int(np.max(counts)), 2)
    norm = colors.Normalize(vmin=1, vmax=vmax)
    facecolors = cmap(norm(counts))
    ax.scatter(unique_coords[:, 0], unique_coords[:, 1], s=8, c=facecolors, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], linewidth=1)
    ax.text(0.02, 0.98, f"loss: {loss:.3f}\nr: {r:.3f}", transform=ax.transAxes, va="top", ha="left", fontsize="small")
    ax.set(title=title, xlabel="output", ylabel="target")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Samples per point", fontsize="small")
    cbar.ax.tick_params(labelsize="x-small")
    return fig
