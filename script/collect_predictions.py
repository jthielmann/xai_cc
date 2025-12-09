import os
from tqdm import tqdm
import pandas as pd
import sys
sys.path.insert(0, '..')
from script.main_utils import parse_yaml_config, read_config_parameter

from torchmetrics.functional import pearson_corrcoef
import torch
def xor(a, b):
    return (a and not b) or (not a and b)

def collect_predictions(geneset):
    model_dirs = []
    base_dir = "../evaluation/predictions/" + geneset
    # single run with best_model.pth and config directly there
    predictions_filename = "predictions.csv"
    for run_name in os.listdir(base_dir):
        run_dir = os.path.join(base_dir, run_name)
        pred_path = os.path.join(run_dir, predictions_filename)
        if os.path.exists(pred_path):
            model_dirs.append((run_dir, run_name))
        else:
            if not os.path.isdir(run_dir):
                continue
            for gene_split_name in os.listdir(run_dir):
                gene_split_dir = os.path.join(run_dir, gene_split_name)
                pred_path = os.path.join(gene_split_dir, predictions_filename)

                if os.path.exists(pred_path):
                    model_dirs.append((gene_split_dir, run_name))

    print(f"Found {len(model_dirs)} models in {base_dir}")

    rows = []
    for model_dir, run_name in tqdm(model_dirs, desc="Runs"):
        df = pd.read_csv(os.path.join(model_dir, predictions_filename))
        preds_cols = [col for col in df.columns if "_pred" in col]
        label_cols = [lab for lab in df.columns if "_label" in lab]
        config_filename = os.path.join("../models/", model_dir[14:], "config")
        config = parse_yaml_config(config_filename)
        loss_fn_switch = read_config_parameter(config, "loss_fn_switch")
        fine_tune_layers = int(read_config_parameter(config, "encoder_finetune_layers"))
        if not read_config_parameter(config, "freeze_encoder"):
            trained_layers = "all layers"
        elif fine_tune_layers > 0:
            trained_layers = f"{fine_tune_layers} enc layers"
        else:
            trained_layers = "heads only"
        for lab in label_cols:
            for pred_col in preds_cols:
                if pred_col[:-5] in lab:
                    preds = torch.tensor(df[pred_col])
                    labs = torch.tensor(df[lab])

                    pearson = pearson_corrcoef(preds, labs)
                    row = [pred_col[:-4], round(pearson.item(), 4), model_dir, run_name, read_config_parameter(config, "loss_fn_switch"), trained_layers]
                    rows.append(row)

    results_path = os.path.join("../evaluation/predictions", geneset)
    os.makedirs(results_path, exist_ok=True)

    results_filename = os.path.join(results_path, predictions_filename)

    print(f"saving to {results_filename}")
    pd.DataFrame(rows, columns=["gene", "pearson", "model_dir", "run_name", "loss", "trained_layers"]).to_csv(results_filename, index=False)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Aggregate predictions into CSV")
    p.add_argument("--geneset", required=True, help="Path to models/<geneset>")
    args = p.parse_args()
    collect_predictions(args.geneset)
