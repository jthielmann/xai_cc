import sys, os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

sys.path.insert(0, "..")

encoder_keys = ["dino", "resnet", "uni"]
def _plotviolin_data(violin_data, geneset, encoder_key=None):
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
    filename = ""

    if encoder_key is not None:
        filename = f"{geneset}_{encoder_key}.svg"
    else:
        filename = f"{geneset}.svg"
    out_path = os.path.join(plot_dir, filename)

    fig.savefig(out_path)
    plt.close(fig)

    return out_path

def plot_violins(geneset):
    base_dir = "../evaluation/predictions/" + geneset
    df_all = pd.read_csv(os.path.join(base_dir, "predictions.csv"))

    violin_jobs = []

    # split the plotting in icms case because there are many runs
    if "icms" in geneset:
        encoder_keys = ["dino", "resnet", "uni"]

        for enc_key in encoder_keys:
            df = df_all[df_all["encoder_type"].str.contains(enc_key, case=False, na=False)]
            if df.empty:
                continue

            violin_data = []
            for run_name in df["run_name"].unique():
                df_run = df[df["run_name"] == run_name]
                pearsons       = df_run["pearson"].values
                loss_fn_switch = df_run["loss"].iloc[0]
                trained_layers = df_run["trained_layers"].iloc[0]
                encoder_type   = df_run["encoder_type"].iloc[0]

                violin_data.append(
                    (run_name, pearsons, loss_fn_switch, trained_layers, encoder_type)
                )

            violin_jobs.append((violin_data, enc_key))

    else:
        df = df_all
        violin_data = []
        for run_name in df["run_name"].unique():
            df_run = df[df["run_name"] == run_name]
            pearsons       = df_run["pearson"].values
            loss_fn_switch = df_run["loss"].iloc[0]
            trained_layers = df_run["trained_layers"].iloc[0]
            encoder_type   = df_run["encoder_type"].iloc[0]

            violin_data.append(
                (run_name, pearsons, loss_fn_switch, trained_layers, encoder_type)
            )

        violin_jobs.append((violin_data, None))

    for violin_data, enc_key in violin_jobs:
        out_path = _plotviolin_data(violin_data, geneset, encoder_key=enc_key)
        print(f"saved violins to {out_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Aggregate predictions into CSV")
    p.add_argument("--geneset", required=True, help="Path to models/<geneset>")
    args = p.parse_args()
    plot_violins(args.geneset)
