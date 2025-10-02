
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import numpy as np
from pathlib import Path

from script.data_processing.data_loader import get_dataset_single_file

def compare_split_dists(config):
    """
    Generates and saves plots and statistics comparing the gene expression
    distributions between training and test sets.

    Args:
        config (dict): A configuration dictionary containing all necessary parameters.
    """
    # 1. Load data using the data loader
    data_dir = config["data_dir"]
    csv_path = config["single_csv_path"]
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(data_dir, csv_path)

    train_df = get_dataset_single_file(
        csv_path=csv_path,
        data_dir=data_dir,
        genes=config["genes"],
        split="train",
        split_col_name=config["split_col_name"],
        tile_subdir=config.get("tile_subdir"),
    ).df

    test_df = get_dataset_single_file(
        csv_path=csv_path,
        data_dir=data_dir,
        genes=config["genes"],
        split="test",
        split_col_name=config["split_col_name"],
        tile_subdir=config.get("tile_subdir"),
    ).df

    genes = config["genes"]
    if not genes:
        # If no genes are specified, use all columns except the tile and split columns
        genes = [col for col in train_df.columns if col not in ["tile", config["split_col_name"]]]


    # 2. Prepare output directory
    save_dir = Path(config.get("save_dir", "docs/gene_dist_comparison"))
    save_dir.mkdir(parents=True, exist_ok=True)

    stats_records = []

    # 3. Generate plots and stats for each gene
    for gene in genes:
        train_values = train_df[gene].dropna()
        test_values = test_df[gene].dropna()

        # a. Overlaid Histograms
        plt.figure(figsize=(10, 6))
        plt.hist(train_values, bins=50, alpha=0.5, label="Train", density=True)
        plt.hist(test_values, bins=50, alpha=0.5, label="Test", density=True)
        plt.title(f"Distribution of {gene} in Train vs. Test Sets")
        plt.xlabel("Gene Expression Value")
        plt.ylabel("Density")
        plt.legend()
        plot_path = save_dir / f"{gene}_dist_comparison.png"
        plt.savefig(plot_path)
        plt.close()

        # b. Summary Statistics
        train_stats = train_values.describe()
        test_stats = test_values.describe()
        
        # c. KS Test
        ks_stat, p_value = ks_2samp(train_values, test_values)

        stats_records.append({
            "gene": gene,
            "train_mean": train_stats["mean"],
            "test_mean": test_stats["mean"],
            "train_std": train_stats["std"],
            "test_std": test_stats["std"],
            "train_min": train_stats["min"],
            "test_min": test_stats["min"],
            "train_max": train_stats["max"],
            "test_max": test_stats["max"],
            "ks_stat": ks_stat,
            "p_value": p_value,
        })

    # 4. Save summary statistics to a Markdown file
    stats_df = pd.DataFrame(stats_records)
    stats_table_path = save_dir / "summary_statistics.md"
    stats_df.to_markdown(stats_table_path, index=False)

    print(f"Comparison plots and statistics saved to {save_dir}")

if __name__ == "__main__":
    # Example configuration, similar to sweeps/configs/coad_single_csv
    # IMPORTANT: Update paths and genes to match your local setup.
    config = {
        "data_dir": "/Users/jona/Documents/code/xai_cc/data/COAD",
        "single_csv_path": "hvg_cmmn_std.csv",
        "split_col_name": "split",
        "genes": ["ENSG00000139618", "ENSG00000148584", "ENSG00000182959"],  # Example genes
        "save_dir": "/Users/jona/Documents/code/xai_cc/docs/gene_dist_comparison",
    }
    
    # Check if the data path exists
    if not os.path.exists(os.path.join(config['data_dir'], config['single_csv_path'])):
        print(f"Data file not found at {os.path.join(config['data_dir'], config['single_csv_path'])}")
        print("Please update the config in script/evaluation/compare_split_dists.py to point to your data.")
    else:
        compare_split_dists(config)
