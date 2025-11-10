import os, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

from script.data_processing.data_loader import get_base_dataset


# assumes your get_base_dataset is already imported

def _pick_gene_data_filename(cfg):
    v = cfg.get("gene_data_filename", None)
    if v is None:
        return "gene_data.csv"
    # During sweeps W&B will pass a single selected value.
    # If you're calling this locally with a list, we pick the first.
    if isinstance(v, (list, tuple)):
        return v[0]
    return v

def plot_data_hists(
    config: dict,
    save_dir: str = None,
    overlay_per_gene: bool = False,
):
    # Accept either flattened configs (preferred) or legacy nested sweep-style
    genes = config.get("genes", None)
    if genes is None:
        p_genes = config.get("parameters", {}).get("genes")
        if isinstance(p_genes, dict):
            genes = p_genes.get("values") or p_genes.get("value")
        else:
            genes = p_genes
    samples = config["train_samples"]
    data_dir = config["data_dir"]
    hist_bins = int(config.get("bins", 50))  # interpret as HISTOGRAM bins
    debug = bool(config.get("debug", False))
    dataset_name = config.get("dataset", "unknown")
    gene_data_filename = _pick_gene_data_filename(config)

    # 1) Build raw (un-resampled) dataframe
    df = get_base_dataset(
        data_dir=data_dir,
        genes=genes,
        samples=samples if samples is not None else [f.name for f in os.scandir(data_dir) if f.is_dir()],
        meta_data_dir=config.get("meta_data_dir", "/meta_data/"),
        max_len=config.get("max_len", None),
        bins=1,  # IMPORTANT: keep raw distribution for histograms
        gene_data_filename=gene_data_filename,
        lds_smoothing_csv=None,  # weights not needed for histograms
        weight_transform=config.get("weight_transform", "inverse"),
        weight_clamp=int(config.get("weight_clamp", 10)),
    )

    # 2) Derive patient column from tile path: .../<data_dir>/<patient>/tiles/<file>
    def _infer_patient_from_tile(p):
        parts = Path(p).parts
        # Expect ... / data_dir / patient / tiles / filename
        # So patient should be the 3rd item from the end.
        return parts[-3] if len(parts) >= 3 else "unknown"
    if "patient" not in df.columns:
        df["patient"] = df["tile"].apply(_infer_patient_from_tile)

    # 3) Optional downsample for speed in debug
    if debug and len(df) > 20000:
        df = df.sample(20000, random_state=0).reset_index(drop=True)

    # 4) W&B init (optional)
    use_wandb = bool(config.get("log_to_wandb", False))
    run = None
    if use_wandb:
        # sanitize config to avoid nested parameters/metric blocks in W&B overview
        wb_config = {k: v for k, v in config.items() if k not in ("parameters", "metric", "method")}
        run = wandb.init(
            project=config.get("project", "histogram"),
            name=config.get("name", f"EDA-{dataset_name}"),
            config=wb_config,
            tags=["eda", f"dataset={dataset_name}", "histograms"],
        )

    # 5) Where to save (optional)
    out_paths = []
    if save_dir:
        save_dir = str(save_dir)
        os.makedirs(save_dir, exist_ok=True)

    # 6) Plot per-patient histograms per gene
    patients = sorted(df["patient"].unique().tolist())
    for g in genes:
        # Overlay plot per gene (all patients in one figure), optional
        if overlay_per_gene:
            fig, ax = plt.subplots()
            for p in patients:
                vals = df.loc[df["patient"] == p, g].dropna().to_numpy()
                if vals.size == 0:
                    continue
                ax.hist(vals, bins=hist_bins, density=True, alpha=0.35, label=p)
            ax.set_title(f"{dataset_name} — {g} (overlay)")
            ax.set_xlabel(g)
            ax.set_ylabel("density")
            if len(patients) <= 12:
                ax.legend(fontsize="x-small", ncol=2)
            if save_dir:
                out = os.path.join(save_dir, f"hist_overlay_{dataset_name}_{g}.png")
                fig.savefig(out, dpi=150, bbox_inches="tight")
                out_paths.append(out)
            if use_wandb:
                wandb.log({f"hists/overlay/{g}": wandb.Image(fig)})
            plt.close(fig)

        # Individual per-patient figures
        for p in patients:
            vals = df.loc[df["patient"] == p, g].dropna().to_numpy()
            if vals.size == 0:
                continue
            fig, ax = plt.subplots()
            ax.hist(vals, bins=hist_bins, density=False)
            ax.set_title(f"{dataset_name} — patient={p} · gene={g}")
            ax.set_xlabel(g)
            ax.set_ylabel("count")

            if save_dir:
                out = os.path.join(save_dir, f"hist_{dataset_name}_{g}_patient_{p}.png")
                fig.savefig(out, dpi=150, bbox_inches="tight")
                out_paths.append(out)
            if use_wandb:
                wandb.log({f"hists/{g}/{p}": wandb.Image(fig)})
            plt.close(fig)

    if use_wandb:
        wandb.summary["n_rows"] = len(df)
        wandb.summary["n_patients"] = len(patients)
        wandb.summary["genes"] = genes
        wandb.finish()

    return out_paths
