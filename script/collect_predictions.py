import os

import pandas as pd
import sys
sys.path.insert(0, '..')
from script.main_utils import read_config_parameter


def xor(a, b):
    return (a and not b) or (not a and b)

def collect_predictions(base_dir):
    model_dirs = []
    model_dirs_incomplete = []
    # single run with best_model.pth and config directly there
    predictions_filename = "predictions/predictions.csv"
    for run_name in os.listdir(base_dir):
        run_dir = os.path.join(base_dir, run_name)
        pred_path = os.path.join(run_dir, predictions_filename)
        if os.path.exists(pred_path):
            model_dirs.append((run_dir, run_name))
            print("single run: ", run_dir)
        else:
            print(f"looping over run_dir {run_dir}", os.listdir(run_dir))
            for gene_split_name in os.listdir(run_dir):
                gene_split_dir = os.path.join(run_dir, gene_split_name)
                pred_path = os.path.join(gene_split_dir, predictions_filename)

                if os.path.exists(pred_path):
                    model_dirs.append((gene_split_dir, run_name))
                    print("split_genes_by: ", gene_split_dir)

    for model_dir in model_dirs:
        df = pd.read_csv(model_dir + predictions_filename)
        preds_cols = [col for col in df.columns if "_pred" in col]
        label_cols = [lab for lab in df.columns if "_label" in lab]
        exit(0)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Aggregate predictions into CSV")
    p.add_argument("--path", required=True, help="Path to models/<geneset>")
    args = p.parse_args()
    path = collect_predictions(args.path)
    print(path)
