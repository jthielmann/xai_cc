import sys, os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

sys.path.insert(0, "..")

def _plotviolin_data(violin_data):
    labels, groups = [], []
    for item in violin_data:
        run_name, vals, loss_fn_switch, trained_layers, encoder_type = item
        arr = np.asarray(vals, dtype=float)
        bad = ~np.isfinite(arr)
        if bad.any():
            idx = np.where(bad)[0][:5].tolist()
            raise ValueError(f"non-finite values for {run_name}; idx={idx}; vals={arr[bad][:5].tolist()}")
        labels.append(encoder_type + "\n" + trained_layers + "\n" + loss_fn_switch)
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
    out_path = os.path.join(out_dir, "violins", ".png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def plot_violins(geneset):
    base_dir = "../evaluation/predictions/" + geneset
    df = pd.read_csv(os.path.join(base_dir, "predictions.csv"))
    violin_data = []
    for run_name in df.run_name.unique():
        pearsons = df[df["run_name"] == run_name]["pearson"].values
        loss_fn_switch = df[df["run_name"] == run_name]["loss_fn_switch"].unique()[0]
        trained_layers = df[df["run_name"] == run_name]["trained_layers"].unique()[0]
        encoder_type = df[df["run_name"] == run_name]["v"].unique()[0]
        violin_data.append((run_name, pearsons, loss_fn_switch, trained_layers, encoder_type))

    out_path = _plotviolin_data(violin_data)
    print(f"saved violins to {out_path}")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Aggregate predictions into CSV")
    p.add_argument("--geneset", required=True, help="Path to models/<geneset>")
    args = p.parse_args()
    plot_violins(args.geneset)