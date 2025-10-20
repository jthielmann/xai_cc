import os
from typing import List

import torch
import pandas as pd
from torch.utils.data import Dataset
from script.data_processing.data_loader import get_dataset
from script.data_processing.image_transforms import get_eval_transforms
import matplotlib.pyplot as plt
import wandb


class _WithTileName(Dataset):
    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        name = self.base.get_tilename(idx)
        return img, y, name


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
):
    if patient is None:
        raise ValueError("Please provide a `patient` (string).")

    for g in genes:
        if not hasattr(model, g):
            raise AttributeError(f"Model has no head named '{g}'")

    model = model.to(device)
    model.eval()
    results_dir = f"../{out_path}/{run_name}/predictions/"
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, "results.csv")
    if os.path.exists(filename):
        raise RuntimeError("output file already exists")
    write_header = not os.path.exists(filename)
    if write_header:
        pd.DataFrame(columns=["labels", "output", "path", "tile", "patient"]).to_csv(
            filename, index=False
        )

    eval_tf = get_eval_transforms(image_size=int(image_size))
    base_ds = get_dataset(
        data_dir=data_dir,
        genes=genes,
        transforms=eval_tf,
        samples=[patient],
        only_inputs=False,
        meta_data_dir=meta_data_dir,
        gene_data_filename=gene_data_filename
    )
    loader = _WithTileName(base_ds)

    outputs_by_gene = {g: None for g in genes}
    handles = []

    def make_hook(gene_name):
        def hook(_module, _inp, out):
            outputs_by_gene[gene_name] = out
        return hook

    for g in genes:
        handles.append(getattr(model, g).register_forward_hook(make_hook(g)))

    i, j = 0, len(loader)
    results = []
    with torch.no_grad():
        for images, labels, name in loader:
            if i % 10 == 0:
                print(i, "/", j)
            i += 1

            for k in outputs_by_gene:
                outputs_by_gene[k] = None

            images = images.unsqueeze(0).to(device).float()

            _ = model(images)

            missing = [g for g, v in outputs_by_gene.items() if v is None]
            if missing:
                raise RuntimeError(f"Missing outputs for genes: {missing}")

            lbl_list = labels.reshape(-1).cpu().tolist()
            out_list = [
                float(outputs_by_gene[g].reshape(-1)[0].cpu().item())
                for g in genes
            ]
            results.append(out_list)

            row = pd.DataFrame({
                "labels": [lbl_list],
                "output": [out_list],
                "path":   [name],
                "tile":   [os.path.basename(name)],
                "patient": patient
            })
            row.to_csv(filename, index=False, mode="a", header=False)

    for h in handles:
        h.remove()

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
