import argparse, os, yaml
import numpy as np, pandas as pd
from lds_helpers import LDS

from script.data_processing.data_loader import get_base_dataset
from script.configs.dataset_config import get_dataset_cfg


def parse_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # flatten the "parameters:" section (keep only scalar .value fields)
    cfg = {k: v for k, v in raw.items() if k != "parameters"}
    for k, entry in raw.get("parameters", {}).items():
        if isinstance(entry, dict) and "value" in entry:   # ignore sweep
            cfg[k] = entry["value"]
    return cfg


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return os.path.abspath(p)


# ──────────────────────────── main logic ────────────────────────────────────
def main(cfg_path: str):
    cfg = parse_yaml_config(cfg_path)

    # --- essentials pulled from cfg (with sane fallbacks) -------------------
    dataset   = cfg.get("dataset",        "crc_base")
    genes     = cfg.get("genes",          ["RUBCNL"])
    bins      = int(cfg.get("lds_bins",   30))
    ks        = int(cfg.get("lds_ks",     5))
    sigma     = cfg.get("lds_sigma")           # None → Silverman
    out_path   = ensure_dir(cfg.get("out_path", "./lds_tables"))

    # dataset metadata (patients list, data dir, …)
    ds_cfg        = get_dataset_cfg(cfg)
    data_dir      = ds_cfg["data_dir"]
    patient_names = ds_cfg["patients"]

    lds = LDS(bins=bins, ks=ks, kind="gaussian", sigma=sigma)

    for gene in genes:
        # 1) collect labels
        df = get_base_dataset(
            data_dir,
            [gene],
            samples=patient_names,
            gene_data_filename=ds_cfg.get("gene_data_file", "gene_data.csv"),
        )
        labels = df[gene].values

        # 2) smooth + divergences
        eff_hist, edges = lds.smooth(labels)
        print(
            f"{gene:>8} |"
            f" JS={lds.compare(labels,'js'):.4f}"
            f" KL={lds.compare(labels,'kl'):.4f}"
            f" WASS={lds.compare(labels,'wass'):.4f}"
        )

        # 3) save artefacts
        np.savez(os.path.join(out_path, f"{gene}_hist_edges.npz"),
                 eff_hist=eff_hist, edges=edges)

        # optional per-tile weights
        idx      = np.clip(np.digitize(labels, edges[:-1]) - 1, 0, bins - 1)
        weights  = 1.0 / np.maximum(eff_hist[idx], 1e-12)
        weightdf = pd.DataFrame({"tile": df["tile"], f"{gene}_lds_w": weights})
        weightdf.to_csv(os.path.join(out_path, f"{gene}_lds_weights.csv"), index=False)

    print("Finished; outputs in →", out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config file")
    main(p.parse_args().config)
