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


def _plot_violin_seaborn(data, x_col, y_col, hue_col, title, out_path, y_lim=(-1, 1)):

    num_categories = len(data[x_col].unique())
    fig_width = max(12, num_categories * 1.2)
    fig_height = 8

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.violinplot(data=data, x=x_col, y=y_col, hue=hue_col, inner=None, cut=0, ax=ax, saturation=0.75)

    sns.stripplot(data=data, x=x_col, y=y_col, hue=hue_col,dodge=True,jitter=0.2,color='black',marker='o',size=2,alpha=0.6,ax=ax,legend=False)

    sns.pointplot(data=data, x=x_col, y=y_col, hue=hue_col, estimator=np.mean, markers='x', linestyles='', dodge=True, color='black', zorder=3, legend=False, ax=ax)

    if hue_col:
        handles, labels = ax.get_legend_handles_labels()
        num_hue_levels = len(data[hue_col].unique())
        ax.legend(
            handles[:num_hue_levels],
            labels[:num_hue_levels],
            title=hue_col,
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
    elif ax.get_legend():
        ax.get_legend().remove()

    ax.set_ylim(*y_lim)
    ax.set_ylabel(y_col)
    ax.set_xlabel("Group")
    ax.set_title(title)

    # Set label rotation and smaller font size
    ax.tick_params(axis='x', rotation=0, labelsize='small')

    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


print("\n--- Starting Violin Plot Generation ---")
saved_paths = []

id_cols = ["encoder_type", "wmse", "encoder_finetune_layers"]

data_map = {
    "icms2up": dfs_icms2up,
    "icms3up": dfs_icms3up,
    "icms2down": dfs_icms2down,
    "icms3down": dfs_icms3down,
    "hvg": hvg,
    "cmmn": cmmn,
}

for set_name, df_list in data_map.items():

    if not df_list:
        print(f"Skipping '{set_name}': No data found.")
        continue

    df_concat = pd.concat(df_list)

    genes = _GENE_SETS.get(set_name, [])
    if not genes:
        print(f"Skipping '{set_name}': No genes found in _GENE_SETS.")
        continue

    pearson_cols = [f"pearson_{g}" for g in genes if f"pearson_{g}" in df_concat.columns]

    if not pearson_cols:
        print(f"Skipping '{set_name}': No matching Pearson columns found in data.")
        continue

    df_long = df_concat.melt(
        id_vars=id_cols,
        value_vars=pearson_cols,
        var_name="gene",
        value_name="Pearson r"
    )

    df_long.dropna(subset=["Pearson r"], inplace=True)

    if df_long.empty:
        print(f"Skipping '{set_name}': No valid Pearson data after melting.")
        continue

    wmse_label = np.where(df_long['wmse'], "wmse", "wmse_off")

    df_long['group_label'] = (
            df_long['encoder_type'].astype(str) + "\n" +
            wmse_label + "\n" +
            "L" + df_long['encoder_finetune_layers'].astype(str)
    )

    title = f"Pearson Correlation for {set_name} genes"
    fname = _sanitize_token(set_name)
    out_path_violin = _ensure_out_path(os.path.join(OUT_DIR, f"{fname}__violin"), "png")

    print(f"Plotting: {title}...")

    _plot_violin_seaborn(
        data=df_long,
        x_col="group_label",
        y_col="Pearson r",
        hue_col=None,
        title=title,
        out_path=out_path_violin
    )

    saved_paths.append(out_path_violin)

print("\n--- Plotting Complete ---")
print(f"Saved {len(saved_paths)} plots to {OUT_DIR}:")
for p in saved_paths:
    print(p)





def plot_violins(geneset):
    base_dir = "../evaluation/predictions/" + geneset
    df = pd.read_csv(os.path.join(base_dir, "predictions.csv"))

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Aggregate predictions into CSV")
    p.add_argument("--geneset", required=True, help="Path to models/<geneset>")
    args = p.parse_args()
    plot_violins(args.geneset)