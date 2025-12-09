import sys, os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

sys.path.insert(0, "..")

# todo: add these:
"""
    sns.stripplot(
        data=data,
        x=x_col,
        y=y_col,
        hue=hue_col,
        dodge=True,
        jitter=0.2,
        color='black',
        marker='o',
        size=2,
        alpha=0.6,
        ax=ax,
        legend=False
    )

    sns.pointplot(
        data=data,
        x=x_col,
        y=y_col,
        hue=hue_col,
        estimator=np.mean,
        markers='x',
        linestyles='',
        dodge=True,
        color='black',
        zorder=3,
        legend=False,
        ax=ax
    )

"""


def _plotviolin_data(violin_data, geneset):
    if not violin_data:
        raise ValueError(f"no violin_data provided; violin_data={violin_data}")
    labels, groups = [], []
    encoder_types = []
    for item in violin_data:
        run_name, vals, loss_fn_switch, trained_layers, encoder_type = item
        encoder_types.append(encoder_type)
        arr = np.asarray(vals, dtype=float)
        bad = ~np.isfinite(arr)
        if bad.any():
            idx = np.where(bad)[0][:5].tolist()
            raise ValueError(f"non-finite values for {run_name}; idx={idx}; vals={arr[bad][:5].tolist()}")
        labels.append(encoder_type + "\n" + trained_layers + "\n" + loss_fn_switch)
        groups.append(arr)
    positions = np.arange(len(labels))
    plot_df = pd.DataFrame({"label": np.repeat(labels, [len(g) for g in groups]), "pearson": np.concatenate(groups)})
    fig_width = max(8, int(1 * len(labels) + 0.99))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ax.violinplot(groups, positions=positions, showmeans=False, showextrema=True, showmedians=True)

    sns.stripplot(data=plot_df, x="label", y="pearson", order=labels, dodge=True, jitter=0.2, color="black", marker="o", size=2, alpha=0.165, ax=ax, legend=False)
    #sns.pointplot(data=plot_df, x="label", y="pearson", order=labels, estimator=np.mean, markers="x", linestyles="", dodge=True, color="black", zorder=3, legend=False, ax=ax, alpha=0.01)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(-.25, 1.0)
    ax.set_ylabel("Pearson r")
    ax.set_xlabel("Run")
    ax.set_title("Pearson by run")
    fig.tight_layout()
    out_dir = "../evaluation"
    plot_dir = os.path.join(out_dir, "violins")
    os.makedirs(plot_dir, exist_ok=True)

    if len(set(encoder_types)) == 1:
        filename = f"{geneset}_{encoder_types[0]}.svg"
    else:
        filename = f"{geneset}.svg"
    out_path = os.path.join(plot_dir, filename)

    fig.savefig(out_path)
    plt.close(fig)

    return out_path

def plot_violins(geneset):
    base_dir = "../evaluation/predictions/" + geneset
    df_all = pd.read_csv(os.path.join(base_dir, "predictions.csv"))
    violin_data_list = []

    if "icms" in geneset:
        encoder_types = ["dinov3", "resnet", "uni"]
        for encoder_type in encoder_types.unique():
            violin_data = []
            df = df_all[df_all["encoder_type"] == encoder_type]

            for run_name in df.run_name.unique():
                pearsons = df[df["run_name"] == run_name]["pearson"].values
                loss_fn_switch = df[df["run_name"] == run_name]["loss"].unique()[0]
                trained_layers = df[df["run_name"] == run_name]["trained_layers"].unique()[0]
                encoder_type = df[df["run_name"] == run_name]["encoder_type"].unique()[0]
                violin_data.append((run_name, pearsons, loss_fn_switch, trained_layers, encoder_type))
            violin_data_list.append(violin_data)
    else:
        violin_data = []
        df = df_all
        for run_name in df.run_name.unique():
            pearsons = df[df["run_name"] == run_name]["pearson"].values
            loss_fn_switch = df[df["run_name"] == run_name]["loss"].unique()[0]
            trained_layers = df[df["run_name"] == run_name]["trained_layers"].unique()[0]
            encoder_type = df[df["run_name"] == run_name]["encoder_type"].unique()[0]
            violin_data.append((run_name, pearsons, loss_fn_switch, trained_layers, encoder_type))
        violin_data_list.append(violin_data)

    for violin_data in violin_data_list:
        out_path = _plotviolin_data(violin_data, geneset)
        print(f"saved violins to {out_path}")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Aggregate predictions into CSV")
    p.add_argument("--geneset", required=True, help="Path to models/<geneset>")
    args = p.parse_args()
    plot_violins(args.geneset)
