import os

import pandas as pd
import sys
sys.path.insert(0, '..')
from torchmetrics.functional import pearson_corrcoef
import torch
def xor(a, b):
    return (a and not b) or (not a and b)

def collect_predictions(base_dir):
    model_dirs = []
    model_dirs_incomplete = []
    # single run with best_model.pth and config directly there
    predictions_filename = "predictions.csv"
    for run_name in os.listdir(base_dir):
        run_dir = os.path.join(base_dir, run_name)
        pred_path = os.path.join(run_dir, predictions_filename)
        if os.path.exists(pred_path):
            model_dirs.append((run_dir, run_name))
        else:
            for gene_split_name in os.listdir(run_dir):
                gene_split_dir = os.path.join(run_dir, gene_split_name)
                pred_path = os.path.join(gene_split_dir, predictions_filename)

                if os.path.exists(pred_path):
                    model_dirs.append((gene_split_dir, run_name))

    print(f"Found {len(model_dirs)} models in {base_dir}")

    rows = []
    for idx, (model_dir, run_name) in enumerate(model_dirs):
        print(idx, f"/{len(model_dirs)} runs")
        if idx == 10:
            break
        df = pd.read_csv(os.path.join(model_dir, predictions_filename))
        preds_cols = [col for col in df.columns if "_pred" in col]
        label_cols = [lab for lab in df.columns if "_label" in lab]
        for lab in label_cols:
            for pred_col in preds_cols:
                if pred_col[:-4] in lab:
                    preds = torch.tensor(df[pred_col])
                    labs = torch.tensor(df[lab])

                    pearson = pearson_corrcoef(preds, labs)
                    row = [pred_col[:-4], pearson.item(), model_dir, run_name]
                    rows.append(row)

    print("rows: ", rows[0])
    exit(0)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Aggregate predictions into CSV")
    p.add_argument("--path", required=True, help="Path to models/<geneset>")
    args = p.parse_args()
    path = collect_predictions(args.path)
    print(path)
