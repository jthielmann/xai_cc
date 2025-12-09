import os, sys, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import wandb
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

sys.path.insert(0, "..")
from script.main_utils import parse_yaml_config, setup_dump_env
from script.boxplot_helpers import (
    _load_forward_metrics_recursive,
    _apply_filters,
    _collect_values_by_encoder,
    _plot_box,
    _plot_violin,
    _sanitize_token,
    _ensure_out_path,
)
from script.evaluation.gather_results import gather_forward_metrics

from script.gene_list_helpers import _GENE_SETS, get_full_gene_list
cfg = {"data_dir": "/data/cephfs-2/unmirrored/groups/krieger/xai/HEST/hest_coad_visium", "meta_data_dir": "metadata", "gene_data_filename": "gene_log1p.csv"}
_GENE_SETS["cmmn"] = get_full_gene_list(cfg)
print(len(_GENE_SETS["cmmn"]))

EVAL_ROOT = "../evaluation"
OUT_DIR = os.path.join(EVAL_ROOT, "boxplots")
setup_dump_env()
df = pd.read_csv("../evaluation/results/forward_metrics.csv")
dfs_icms2up = []
dfs_icms3up = []
dfs_icms2down = []
dfs_icms3down = []
hvg = []
cmmn = []
for wmse in df["wmse"].unique():
    for encoder_type in df["encoder_type"].unique():
        for encoder_finetune_layers in df["encoder_finetune_layers"].unique():
            masked_icms2up = df[(df["run_name"].str.contains("icms2up")) & (df["wmse"] == wmse) & (
                        df["encoder_type"] == encoder_type) & (
                                            df["encoder_finetune_layers"] == encoder_finetune_layers)]
            masked_icms3up = df[
                df["run_name"].str.contains("icms3up") & (df["wmse"] == wmse) & (df["encoder_type"] == encoder_type) & (
                            df["encoder_finetune_layers"] == encoder_finetune_layers)]
            masked_icms2down = df[df["run_name"].str.contains("icms2down") & (df["wmse"] == wmse) & (
                        df["encoder_type"] == encoder_type) & (
                                              df["encoder_finetune_layers"] == encoder_finetune_layers)]
            masked_icms3down = df[df["run_name"].str.contains("icms3down") & (df["wmse"] == wmse) & (
                        df["encoder_type"] == encoder_type) & (
                                              df["encoder_finetune_layers"] == encoder_finetune_layers)]
            masked_hvg = df[
                df["run_name"].str.contains("hvg") & (df["wmse"] == wmse) & (df["encoder_type"] == encoder_type) & (
                            df["encoder_finetune_layers"] == encoder_finetune_layers)]
            masked_cmmn = df[
                df["run_name"].str.contains("cmmn") & (df["wmse"] == wmse) & (df["encoder_type"] == encoder_type) & (
                            df["encoder_finetune_layers"] == encoder_finetune_layers)]
            if not masked_icms2up.empty:
                dfs_icms2up.append(masked_icms2up)
            if not masked_icms3up.empty:
                dfs_icms3up.append(masked_icms3up)
            if not masked_icms2down.empty:
                dfs_icms2down.append(masked_icms2down)
            if not masked_icms3down.empty:
                dfs_icms3down.append(masked_icms3down)
            if not masked_hvg.empty:
                hvg.append(masked_hvg)
            if not masked_cmmn.empty:
                cmmn.append(masked_cmmn)

print("len(dfs_icms2up) ", len(dfs_icms2up))
print("len(dfs_icms3up) ", len(dfs_icms3up))
print("len(dfs_icms2down) ", len(dfs_icms2down))
print("len(dfs_icms3down) ", len(dfs_icms3down))
print("len(hvg) ", len(hvg))
print("len(cmmn) ", len(cmmn))
pearson_cols_dfs_icms2up = [f"pearson_{g}" for g in _GENE_SETS["icms2up"] for df_i in dfs_icms2up if
                            f"pearson_{g}" in df_i.columns and df_i[f"pearson_{g}"].notna().any()]
pearson_cols_dfs_icms3up = [f"pearson_{g}" for g in _GENE_SETS["icms3up"] for df_i in dfs_icms3up if
                            f"pearson_{g}" in df_i.columns and df_i[f"pearson_{g}"].notna().any()]
pearson_cols_dfs_icms2down = [f"pearson_{g}" for g in _GENE_SETS["icms2down"] for df_i in dfs_icms2down if
                              f"pearson_{g}" in df_i.columns and df_i[f"pearson_{g}"].notna().any()]
pearson_cols_dfs_icms3down = [f"pearson_{g}" for g in _GENE_SETS["icms3down"] for df_i in dfs_icms3down if
                              f"pearson_{g}" in df_i.columns and df_i[f"pearson_{g}"].notna().any()]
pearson_cols_dfs_hvg = [f"pearson_{g}" for g in _GENE_SETS["hvg"] for df_i in hvg if
                        f"pearson_{g}" in df_i.columns and df_i[f"pearson_{g}"].notna().any()]
pearson_cols_dfs_cmmn = [f"pearson_{g}" for g in _GENE_SETS["cmmn"] for df_i in cmmn if
                         f"pearson_{g}" in df_i.columns and df_i[f"pearson_{g}"].notna().any()]
print("len(pearson_cols_dfs_icms2up) ", len(set(pearson_cols_dfs_icms2up)))
print("len(pearson_cols_dfs_icms3up) ", len(set(pearson_cols_dfs_icms3up)))
print("len(pearson_cols_dfs_icms2down) ", len(set(pearson_cols_dfs_icms2down)))
print("len(pearson_cols_dfs_icms3down) ", len(set(pearson_cols_dfs_icms3down)))
print("len(pearson_cols_dfs_hvg) ", len(set(pearson_cols_dfs_hvg)))
print("len(pearson_cols_dfs_cmmn) ", len(set(pearson_cols_dfs_cmmn)))
print(set(pearson_cols_dfs_icms2up))


def _plotviolin_data(violin_data):
    labels, groups = [], []
    for item in violin_data:
        run_name, vals = item
        arr = np.asarray(vals, dtype=float)
        bad = ~np.isfinite(arr)
        if bad.any():
            idx = np.where(bad)[0][:5].tolist()
            raise ValueError(f"non-finite values for {run_name}; idx={idx}; vals={arr[bad][:5].tolist()}")
        labels.append(run_name)
        groups.append(arr)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.violinplot(groups, showmeans=False, showextrema=True, showmedians=False)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Pearson r")
    ax.set_xlabel("Run")
    ax.set_title("Pearson by run")
    fig.tight_layout()
    out_dir = "../evaluation/debug/violins"
    os.makedirs(out_dir, exist_ok=True)
    out_path = _ensure_out_path(os.path.join(out_dir, "violins"), "png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def plot_violins(geneset):
    base_dir = "../evaluation/predictions/" + geneset
    df = pd.read_csv(os.path.join(base_dir, "predictions.csv"))
    violin_data = []
    for run_name in df.run_name.unique():
        pearsons = df[df["run_name"] == run_name]["pearson"].values()
        violin_data.append((run_name, pearsons))

    out_path = _plotviolin_data(violin_data)
    print(f"saved violins to {out_path}")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Aggregate predictions into CSV")
    p.add_argument("--geneset", required=True, help="Path to models/<geneset>")
    args = p.parse_args()
    plot_violins(args.geneset)