import os
from typing import List

import torch
import pandas as pd
from torch.utils.data import Dataset
from script.data_processing.data_loader import get_dataset
from script.data_processing.image_transforms import get_eval_transforms
import matplotlib.pyplot as plt
import wandb


def generate_results(
    model,
    device,
    data_dir,
    genes: List[str],
    run_name,
    out_path,
    meta_data_dir,
    gene_data_filename,
    patient=None,
    make_hists=False,
    wandb_run=None,
    image_size: int = 224,
    max_len: int | None = None,
    forward_batch_size: int = 32,
    forward_num_workers: int = 0,
):
    if patient is None:
        raise ValueError("Please provide a `patient` (string).")

    for g in genes:
        if not hasattr(model, g):
            raise AttributeError(f"Model has no head named '{g}'")

    model = model.to(device)
    model.eval()
    # Respect the configured out_path verbatim; don't prepend "../" here
    results_dir = os.path.join(out_path, run_name, "predictions")
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, "results.csv")
    # If the file already exists, we append to it; otherwise create with header once
    write_header = not os.path.exists(filename)
    columns = []
    for g in genes:
        columns.append(f"label_{g}")
        columns.append(f"pred_{g}")
    columns.append("path")
    columns.append("tile")
    columns.append("patient")


    if write_header:
        pd.DataFrame(columns=columns).to_csv(
            filename, index=False
        )

    eval_tf = get_eval_transforms(image_size=int(image_size))
    ds = get_dataset(
        data_dir=data_dir,
        genes=genes,
        transforms=eval_tf,
        samples=[patient],
        max_len=max_len,
        only_inputs=False,
        meta_data_dir=meta_data_dir,
        gene_data_filename=gene_data_filename,
        return_patient_and_tilepath=True
    )
    # Batched DataLoader; no fallbacks
    from torch.utils.data import DataLoader
    pin = isinstance(device, torch.device) and device.type == "cuda"
    loader = DataLoader(
        ds,
        batch_size=int(forward_batch_size),
        shuffle=False,
        num_workers=int(forward_num_workers),
        pin_memory=pin,
    )

    if not hasattr(model, "gene_to_idx") or not isinstance(model.gene_to_idx, dict):
        raise RuntimeError("Model must expose 'gene_to_idx' mapping for batched inference.")
    try:
        gene_indices = [int(model.gene_to_idx[g]) for g in genes]
    except Exception as e:
        raise RuntimeError(f"Requested genes are not present in model mapping: {genes}") from e

    processed = 0
    total = len(ds)
    with torch.no_grad():
        for images, labels, patients_b, names_b in loader:
            images = images.to(device).float()
            y_hat = model(images)
            # Align and move to CPU for writing
            y_hat = y_hat[:, gene_indices].detach().cpu().float()
            y_true = torch.as_tensor(labels).detach().cpu().float()
            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(1)

            rows = []
            B = int(y_hat.size(0))
            for bi in range(B):
                row = {}
                yh = y_hat[bi].tolist()
                yt = y_true[bi].tolist()
                for gi, g in enumerate(genes):
                    row[f"label_{g}"] = float(yt[gi])
                    row[f"pred_{g}"] = float(yh[gi])
                path = names_b[bi]
                pat = patients_b[bi]
                row["path"] = path
                row["tile"] = os.path.basename(path)
                row["patient"] = pat
                rows.append(row)

            if rows:
                pd.DataFrame(rows).to_csv(filename, index=False, mode="a", header=False)

            processed += B
            # Print frequent progress; first print happens after first batch
            print(f"{processed} / {total}")

    print(f"[generate_results] Saved results to: {filename}")

    if make_hists:
        if not wandb_run:
            raise RuntimeError(f"make_hists {make_hists} but wandb_run is None")
        # TODO
    return filename


def log_patient_hist_from_csv(results_csv: str, data_dir: str, run_name, patients: List[str], genes: List[str], wandb_run):
    hist_dir = f"../evaluation/{run_name}/hists/"
    os.makedirs(hist_dir, exist_ok=True)
    for patient in patients:

        results_dir = f"{hist_dir}/{patient}/"
        meta_dir = config.get("meta_data_dir")
        gene_fn = config.get("gene_data_filename")
        orig_csv = os.path.join(data_dir, patient, meta_dir, gene_fn)

        try:
            orig_df = pd.read_csv(orig_csv, usecols=[gene])
        except Exception as e:
            print(f"[EvalPipeline] Could not load original labels for patient={patient}: {e}")
            return

        try:
            pred_df = pd.read_csv(results_csv, usecols=["output"])
        except Exception as e:
            print(f"[EvalPipeline] Could not load predictions from {results_csv}: {e}")
            return

        orig_vals = pd.to_numeric(orig_df[gene], errors="coerce").to_numpy()
        pred_vals = pd.to_numeric(pred_df[gene], errors="coerce").to_numpy()

        plt.figure()
        plt.hist(orig_df, bins=50, density=True, alpha=0.5, label="original")
        plt.hist(pred_df, bins=50, density=True, alpha=0.5, label="predictions")
        plt.title(f"Pred vs Original — patient={patient} · gene={gene}")
        plt.xlabel(gene)
        plt.ylabel("density")
        plt.legend()


        img_path = os.path.join(hist_dir, f"hist_{gene}_{patient}.png")
        try:
            plt.savefig(img_path, dpi=150, bbox_inches="tight")
        finally:
            plt.close()

        if wandb_run:
            try:
                wandb_run.log({
                    f"hist_img/{patient}/{gene}": wandb.Image(img_path),
                    f"hist_data/{patient}/{gene}/original": wandb.Histogram(orig_vals),
                    f"hist_data/{patient}/{gene}/predictions": wandb.Histogram(pred_vals),
                })
            except Exception as e:
                print(f"[EvalPipeline] W&B logging failed: {e}")

        print(f"[EvalPipeline] Saved histogram: {img_path}")
