import gc
import os
import hashlib
from typing import Any, Dict
import numpy as np
import torch
from umap import UMAP
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import yaml

#from script.evaluation.crp_plotting import plot_crp, plot_crp_zennit, plot_crp2
from script.evaluation.lrp_plotting import plot_lrp, plot_lrp_custom
from script.evaluation.lxt_plotting import plot_lxt
from script.evaluation.scatter_plotting import plot_scatter
from script.evaluation.generate_results import generate_results
from script.evaluation.gather_results import gather_forward_metrics, _compute_metrics_for_csv, _weighted_mean
from script.model.model_factory import get_encoder
from script.model.lit_model import load_lit_regressor
from script.model.encoder_detection import detect_encoder_family
from script.data_processing.data_loader import get_dataset_from_config, get_dataset
from script.data_processing.image_transforms import get_transforms, get_eval_transforms
from torch.utils.data import DataLoader, Subset
"""
Evaluation pipeline for non-CRP cases (LRP, LXT, scatter, UMAP, triptych, forward_to_csv, SAE).

Note: eval_main.py already prepares/merges the config and ensures eval_path. Do NOT call
prepare_cfg here to avoid double-prep and redundant writes.
"""
from script.evaluation.eval_helpers import (
    auto_device,
    collect_state_dicts,
)
from script.evaluation.tri_plotting import plot_triptych_from_model
from script.main_utils import compute_genes_id

def update_forward_metrics_global(eval_path: str, results_csv: str, project: str,
                                  run_name: str, model_path: str, debug: bool) -> str:
    if not eval_path:
        raise ValueError("eval_path required")
    if not results_csv or not os.path.isfile(results_csv):
        raise FileNotFoundError(f"results_csv missing: {results_csv}")
    if not project or not run_name or not model_path:
        raise ValueError("project, run_name, model_path required")
    if debug:
        return ""
    p = os.path.abspath(eval_path)
    while os.path.basename(p) != "evaluation":
        np_ = os.path.dirname(p)
        if np_ == p:
            raise RuntimeError(f"cannot locate evaluation root from eval_path: {eval_path}")
        p = np_
    agg_dir = os.path.join(p, "results")
    os.makedirs(agg_dir, exist_ok=True)
    agg_csv = os.path.join(agg_dir, "forward_metrics.csv")
    metrics, counts = _compute_metrics_for_csv(results_csv)
    pearson_vals = {k[8:]: v for k, v in metrics.items() if k.startswith("pearson_")}
    mse_vals = {k[4:]: v for k, v in metrics.items() if k.startswith("mse_")}
    cfg_path = os.path.join(eval_path, "config")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"missing eval config at {cfg_path}")
    with open(cfg_path, "r") as f:
        eval_cfg = yaml.safe_load(f)
    if not isinstance(eval_cfg, dict):
        raise RuntimeError("eval config invalid; expected mapping")
    mc = eval_cfg.get("model_config")
    if not isinstance(mc, dict):
        raise RuntimeError("eval config missing model_config")
    enc = mc.get("encoder_type")
    if not isinstance(enc, str) or not enc.strip():
        raise RuntimeError("encoder_type missing in model config")
    loss_name = mc.get("loss_fn_switch")
    freeze_encoder_cfg = bool(mc.get("freeze_encoder", False))
    efl = int(mc.get("encoder_finetune_layers", 0) or 0)
    wmse_flag = str(loss_name).strip().lower() in {"wmse", "weighted mse"}
    row = {
        "encoder_type": str(enc),
        "project": str(project),
        "run_name": str(run_name),
        "model_path": str(model_path),
        "metric_name": "mse",
        "loss_name": loss_name,
        "wmse": bool(wmse_flag),
        "freeze_encoder": bool(freeze_encoder_cfg),
        "encoder_finetune_layers": int(efl),
        "pearson_mean": _weighted_mean(pearson_vals, counts),
        "mse_mean": _weighted_mean(mse_vals, counts),
    }
    row.update(metrics)
    if os.path.exists(agg_csv):
        d = pd.read_csv(agg_csv)
        mask = (
            (d.get("project", pd.Series(dtype=str)).astype(str) == row["project"]) &
            (d.get("run_name", pd.Series(dtype=str)).astype(str) == row["run_name"]) &
            (d.get("model_path", pd.Series(dtype=str)).astype(str) == row["model_path"]) 
        )
        idx = np.where(mask.to_numpy() if hasattr(mask, 'to_numpy') else mask.values)[0]
        if idx.size > 0:
            i = int(idx[0])
            old = d.loc[i, "mse_mean"] if "mse_mean" in d.columns else float("nan")
            new = row.get("mse_mean", float("nan"))
            better = (not np.isfinite(old) and np.isfinite(new)) or (np.isfinite(old) and np.isfinite(new) and new < float(old))
            if better:
                for k, v in row.items():
                    d.loc[i, k] = v
                d.to_csv(agg_csv, index=False)
            return agg_csv
        d = pd.concat([d, pd.DataFrame([row])], ignore_index=True)
        d.to_csv(agg_csv, index=False)
        return agg_csv
    pd.DataFrame([row]).to_csv(agg_csv, index=False)
    return agg_csv

def append_geneset_predictions_global(eval_path: str, results_csv: str, cfg: Dict[str, Any],
                                      run_name: str, epsilon: float = 0.1) -> str:
    if not eval_path:
        raise ValueError("eval_path required")
    if not results_csv or not os.path.isfile(results_csv):
        raise FileNotFoundError(f"results_csv missing: {results_csv}")
    if bool(cfg.get("debug", False)):
        return ""
    p = os.path.abspath(eval_path)
    while os.path.basename(p) != "evaluation":
        np_ = os.path.dirname(p)
        if np_ == p:
            raise RuntimeError(f"cannot locate evaluation root from eval_path: {eval_path}")
        p = np_
    out_dir = os.path.join(p, "results", "genesets")
    os.makedirs(out_dir, exist_ok=True)
    

    mc = (cfg or {}).get("model_config") or {}
    enc = mc.get("encoder_type")
    if not isinstance(enc, str) or not enc.strip():
        raise RuntimeError("encoder_type missing in model config")
    project = cfg.get("project") or mc.get("project")
    if not project:
        raise RuntimeError("project missing in config for predictions aggregation")
    loss_name = mc.get("loss_fn_switch")
    wmse_flag = str(loss_name).strip().lower() in {"wmse", "weighted mse"}
    freeze_encoder_cfg = bool(mc.get("freeze_encoder", False))
    efl = int(mc.get("encoder_finetune_layers", 0) or 0)
    gene_set_val = cfg.get("gene_set")
    if not isinstance(gene_set_val, str) or not gene_set_val.strip():
        raise RuntimeError(f"gene_set missing in eval config; cfg_keys={cfg.keys()}")
    gene_set = gene_set_val
    genes = mc.get("genes")
    if genes is None:
        raise RuntimeError("model_config.genes missing; required for genes_id")
    genes_id = compute_genes_id(genes)
    model_path = cfg.get("model_state_path") or ""

    df = pd.read_csv(results_csv)
    if "path" not in df.columns or "patient" not in df.columns:
        raise RuntimeError("results.csv missing required columns 'path' and 'patient'")
    df = df.rename(columns={"path": "tile_path"})
    df["encoder_type"] = str(enc)
    df["loss_name"] = loss_name
    df["project"] = str(project)
    df["run_name"] = str(run_name)
    df["model_path"] = str(model_path)
    df["gene_set"] = gene_set
    df["genes_id"] = genes_id
    df["wmse"] = bool(wmse_flag)
    df["freeze_encoder"] = bool(freeze_encoder_cfg)
    df["encoder_finetune_layers"] = int(efl)

    pk = [
        "project",
        "gene_set",
        "model_path",
        "run_name",
        "patient",
        "tile_path",
        "wmse",
        "encoder_finetune_layers",
        "freeze_encoder",
    ]

    def _coerce_schema(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        str_cols = [
            "project", "gene_set", "model_path", "run_name", "patient", "tile_path",
            "encoder_type", "loss_name", "genes_id",
        ]
        for c in str_cols:
            if c in df_out.columns:
                df_out[c] = df_out[c].astype(str)
        if "encoder_finetune_layers" in df_out.columns:
            df_out["encoder_finetune_layers"] = (
                pd.to_numeric(df_out["encoder_finetune_layers"], errors="coerce").fillna(0).astype(int)
            )
        def _as_bool(sr: pd.Series) -> pd.Series:
            return sr.map(lambda x: str(x).strip().lower() in {"1","true","t","yes","y"})
        for c in ("wmse", "freeze_encoder"):
            if c in df_out.columns:
                df_out[c] = _as_bool(df_out[c])
        return df_out

    def _append_with_conflict_check(target_csv: str, batch_df_raw: pd.DataFrame) -> str:
        batch_df = _coerce_schema(batch_df_raw)
        if os.path.exists(target_csv):
            cur = pd.read_csv(target_csv, low_memory=False)
            cur = _coerce_schema(cur)
            shared = [c for c in cur.columns if c in batch_df.columns]
            if not shared:
                out = pd.concat([cur, batch_df], ignore_index=True, sort=False)
                out.to_csv(target_csv, index=False)
                return target_csv
            j = cur[shared].merge(batch_df[shared], on=pk, how="inner", suffixes=("_old", "_new"))
            num_cols = [
                c for c in shared
                if c not in pk and (c.startswith("pred_") or c.startswith("label_"))
            ]
            if num_cols and not j.empty:
                for c in num_cols:
                    co = f"{c}_old"
                    cn = f"{c}_new"
                    if co in j.columns and cn in j.columns:
                        xo = pd.to_numeric(j[co], errors="coerce")
                        xn = pd.to_numeric(j[cn], errors="coerce")
                        both_nan = xo.isna() & xn.isna()
                        diff = (xo - xn).abs()
                        conflict = (~both_nan) & diff.gt(float(epsilon))
                        if bool(conflict.any()):
                            raise RuntimeError(
                                f"predictions.csv conflict for key={pk}; column={c}; epsilon={epsilon}"
                            )
            cur_keys = cur.set_index(pk).index
            new_keys = batch_df.set_index(pk).index
            cur_keep = cur.loc[~cur_keys.isin(new_keys)].copy()
            out = pd.concat([cur_keep, batch_df], ignore_index=True, sort=False)
            out.to_csv(target_csv, index=False)
            return target_csv
        batch_df.to_csv(target_csv, index=False)
        return target_csv

    gene_csv_path = os.path.join(out_dir, f"{gene_set}.csv")
    return _append_with_conflict_check(gene_csv_path, df)

class EvalPipeline:
    def __init__(self, config, run):
        self.config = dict(config)
        self.wandb_run = run
        self.run_name = self.config.get("run_name")
        self.model_src = self.config.get("model_config_path")
        self.model = self._load_model()
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            raise RuntimeError("No GPU backend available (CUDA/MPS). Please run on a GPU machine.")
        self.model = self.model.to(device)
        # Derive a folder name from provided state paths (preferred over run_name)
        self.model_name = self._derive_model_name()
        os.makedirs(os.path.join(self.config["eval_path"], self.model_name), exist_ok=True)

        # Auto-enable XAI cases based on the actual encoder family when in auto mode
        # Do not disable anything explicitly set by the user.
        if str(self.config.get("xai_pipeline", "manual")).strip().lower() == "auto":
            enc = getattr(self.model, "encoder", self.model)
            fam = detect_encoder_family(enc)
            if fam == "vit" and not bool(self.config.get("lxt", False)):
                self.config["lxt"] = True
            if fam == "resnet" and not bool(self.config.get("lrp", False)):
                self.config["lrp"] = True

        # Enforce: always use the model's genes; never accept 'genes' in eval config.
        model_cfg = self.config.get("model_config")
        if "genes" in self.config and self.config["genes"] is not None:
            raise ValueError(
                "Eval config must not set 'genes'. The trained model's genes are always used. "
                "Remove 'genes' from the eval config."
            )
        mc_genes = model_cfg.get("genes")
        if not mc_genes:
            raise ValueError("Trained model config does not contain 'genes'; cannot proceed with evaluation.")
        # Set genes from model config (exclude empty/unnamed)
        valid_genes = []
        for g in mc_genes:
            gs = str(g).strip()
            if not gs:
                continue
            if gs.lower().startswith("unnamed"):
                continue
            valid_genes.append(gs)
        if not valid_genes:
            raise ValueError("No valid genes after excluding empty/unnamed entries.")
        self.config["genes"] = valid_genes
        # Defaults for single-gene cases
        if not self.config.get("gene") and self.config.get("genes"):
            self.config["gene"] = str(self.config["genes"][0])
        # Provide a default patient (first test sample) for cases that need one
        if not self.config.get("patient"):
            test_samples = self.config.get("test_samples") or model_cfg.get("test_samples")
            if isinstance(test_samples, (list, tuple)) and len(test_samples) > 0:
                self.config["patient"] = str(test_samples[0])

    def cleanup(self):
        model = getattr(self, "model", None)
        if model is not None:
            model.cpu()
        self.model = None
        self.config = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
        gc.collect()

    def _load_model(self):
        state_dicts = collect_state_dicts(self.config)
        if isinstance(self.config.get("model_config"), dict):
            self.config["model_config"]["use_fds"] = False
        return load_lit_regressor(self.config["model_config"], state_dicts)

    def _derive_model_name(self) -> str:
        return ""

    def run(self):
        if self.config.get("lrp"):
            lrp_backend = str(self.config.get("lrp_backend", "zennit")).lower()
            eval_tf = get_transforms(self.config["model_config"], split="eval")
            debug = bool(self.config.get("debug", False))
            cfg = self.config
            meta_dir = cfg.get("meta_data_dir")
            if not meta_dir:
                raise ValueError("meta_data_dir missing; dataset config must provide it")
            gene_csv = cfg.get("gene_data_filename")
            if not gene_csv:
                raise ValueError("gene_data_filename missing; dataset config must provide it")
            data_dir = self.config.get("data_dir")
            if not data_dir:
                raise ValueError("Missing 'data_dir' in eval config for LRP; set config.data_dir.")
            ds = get_dataset(
                data_dir=data_dir,
                genes=None,
                transforms=eval_tf,
                samples=self.config.get("test_samples"),
                max_len=self.config.get("max_len") if debug else None,
                only_inputs=False,
                gene_data_filename=gene_csv,
                meta_data_dir=meta_dir,
                return_patient_and_tilepath=True,
            )
            n = min(int(self.config.get("lrp_max_items", 10)), len(ds))
            loader = DataLoader(Subset(ds, list(range(n))), batch_size=1, shuffle=False)
            # Prepare a local save dir for LRP outputs regardless of W&B.
            lrp_dir = os.path.join(self.config["eval_path"], self.model_name, "lrp")
            os.makedirs(lrp_dir, exist_ok=True)
            print(f"[Eval] Starting LRP (backend={lrp_backend}) on {n} items -> {lrp_dir}")
            if lrp_backend == "custom":
                plot_lrp_custom(self.model, loader, run=self.wandb_run, out_path=lrp_dir)
            else:
                plot_lrp(self.model, loader, run=self.wandb_run, out_path=lrp_dir)

        # Note: PCX has been moved to the CRP pipeline (script/crp_main.py).
        if self.config.get("diff"):
            # Generate spatial triptychs directly from the model and dataset — no merge.csv required.
            # Iterate over dataset-configured samples by default to avoid stale/hardcoded patients.
            out_dir = os.path.join(self.config["eval_path"], self.model_name, "diff")
            os.makedirs(out_dir, exist_ok=True)

            # Prefer explicit 'patients' list, else use dataset-configured test_samples.
            # Fall back to a single 'patient' only if no list is available.
            patients_cfg = self.config.get("patients")
            if isinstance(patients_cfg, (list, tuple)) and len(patients_cfg) > 0:
                patients = list(patients_cfg)
            else:
                patients = list(self.config.get("test_samples", []))
                if not patients:
                    p_single = self.config.get("patient")
                    if p_single:
                        patients = [str(p_single)]

            # De-duplicate while preserving order
            seen = set()
            patients = [p for p in patients if not (p in seen or seen.add(p))]

            gene = self.config["gene"]
            max_items = int(self.config.get("diff_max_items", 0) or 0)
            for p in patients:
                print(f"[Eval] Starting diff triptych for patient={p}, gene={gene} -> {out_dir}")
                plot_triptych_from_model(
                    self.model,
                    self.config,
                    p,
                    gene,
                    out_dir,
                    max_items=max_items if max_items > 0 else None,
                    is_online=bool(self.wandb_run),
                    wandb_run=self.wandb_run,
                )

        if self.config.get("scatter"):
            cfg_scatter = dict(self.config)
            scatter_dir = os.path.join(self.config["eval_path"], self.model_name, "scatter")
            cfg_scatter["out_path"] = scatter_dir
            print(f"[Eval] Starting scatter plots -> {cfg_scatter['out_path']}")
            plot_scatter(
                cfg_scatter,
                self.model,
                wandb_run=self.wandb_run,
                out_path=scatter_dir,
            )

        if self.config.get("umap"):
            # Generate UMAP visualizations from a specified model layer
            layer = self.config.get("umap_layer")
            if not layer or not isinstance(layer, str):
                raise ValueError("'umap' is enabled, but 'umap_layer' is missing. Provide a dot-path layer name.")

            umap_dir = os.path.join(self.config["eval_path"], self.model_name, "umap")
            os.makedirs(umap_dir, exist_ok=True)

            # Build eval loader (inputs only)
            eval_tf = get_transforms(self.config["model_config"], split="eval")
            debug = bool(self.config.get("debug", False))
            cfg = self.config
            meta_dir = cfg.get("meta_data_dir")
            if not meta_dir:
                raise ValueError("meta_data_dir missing; dataset config must provide it")
            gene_csv = cfg.get("gene_data_filename")
            if not gene_csv:
                raise ValueError("gene_data_filename missing; dataset config must provide it")
            ds = get_dataset_from_config(
                dataset_name=self.config["dataset"],
                genes=None,
                split="test",
                debug=debug,
                transforms=eval_tf,
                samples=self.config.get("test_samples"),
                max_len=self.config.get("max_len") if debug else None,
                only_inputs=True,
                meta_data_dir=meta_dir,
                gene_data_filename=gene_csv,
            )
            bs = int(self.config.get("umap_batch_size", 32))
            loader = DataLoader(ds, batch_size=bs, shuffle=False)

            device = auto_device(self.model)
            self.model.eval()
            print(f"[Eval] Starting UMAP for layer='{layer}' -> {umap_dir}")

            def _as_2d(t: torch.Tensor) -> torch.Tensor:
                if isinstance(t, (list, tuple)) and len(t) > 0:
                    t = t[0]
                if hasattr(t, "last_hidden_state"):
                    t = t.last_hidden_state
                if t.ndim == 4:
                    return torch.flatten(t, 1)
                if t.ndim == 3:
                    return t.mean(dim=1)
                if t.ndim == 2:
                    return t
                return torch.flatten(t, 1)

            features = []
            max_samples = self.config.get("umap_max_samples")

            if layer == "encoder":
                # Use encoder features directly for UMAP when layer == "encoder".
                with torch.no_grad():
                    for imgs in loader:
                        imgs = imgs.to(device)
                        z = self.model.encoder(imgs)
                        z = _as_2d(z)
                        features.append(z.detach().cpu().numpy())
                        if max_samples is not None and sum(f.shape[0] for f in features) >= max_samples:
                            break
            else:
                # Resolve a nested module by dot path and hook its output
                target: torch.nn.Module = self.model
                for token in layer.split('.'):
                    if not hasattr(target, token):
                        raise AttributeError(f"umap_layer '{layer}' not found on model at '{token}'")
                    target = getattr(target, token)
                    if not isinstance(target, torch.nn.Module):
                        raise AttributeError(f"Path '{layer}' does not resolve to a nn.Module (stopped at '{token}')")

                captured = {}

                def _hook(_m, _inp, _out):
                    captured["t"] = _out

                handle = target.register_forward_hook(_hook)
                try:
                    with torch.no_grad():
                        for imgs in loader:
                            imgs = imgs.to(device)
                            _ = self.model(imgs)
                            t = captured.get("t")
                            if t is None:
                                raise RuntimeError(f"Hook on '{layer}' did not capture any output")
                            t2d = _as_2d(t)
                            features.append(t2d.detach().cpu().numpy())
                            captured.clear()
                            if max_samples is not None and sum(f.shape[0] for f in features) >= max_samples:
                                break
                finally:
                    handle.remove()

            if not features:
                raise RuntimeError("No features captured for UMAP")

            X = np.concatenate(features, axis=0)
            if max_samples is not None and X.shape[0] > max_samples:
                X = X[:max_samples]
            patient_ids = ds.df['patient'].tolist()
            if max_samples is not None and len(patient_ids) > max_samples:
                patient_ids = patient_ids[:max_samples]
            patients = sorted(list(set(patient_ids)))
            colors = {p: i for i, p in enumerate(patients)}

            # UMAP sweep
            sweep = self.config.get("umap_sweep_params", {"n_neighbors": [15], "min_dist": [0.1]})
            n_neighbors_list = sweep.get("n_neighbors", [15])
            min_dist_list = sweep.get("min_dist", [0.1])
            n_components = int(self.config.get("umap_n_components", 2))

            umap_table_rows = []
            for nn_ in n_neighbors_list:
                for md in min_dist_list:
                    print(f"[Eval]   UMAP fit (n_neighbors={nn_}, min_dist={md})")
                    umap = UMAP(n_components=n_components, n_neighbors=int(nn_), min_dist=float(md), random_state=42, n_jobs=1)
                    emb = umap.fit_transform(X)
                    plt.figure(figsize=(12, 9))
                    for p in patients:
                        idx = [i for i, pid in enumerate(patient_ids) if pid == p and i < emb.shape[0]]
                        if not idx:
                            continue
                        plt.scatter(emb[idx, 0], emb[idx, 1], s=12, label=p)
                    plt.title(f"UMAP layer={layer} (nn={nn_}, md={md})")
                    plt.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout(rect=[0, 0, 0.85, 1])
                    fn = f"umap_layer={layer.replace('.', '_')}_nn={nn_}_md={md}.png"
                    out_path = os.path.join(umap_dir, fn)
                    plt.savefig(out_path, dpi=200)
                    # Add to W&B table if online
                    if self.wandb_run is not None:
                        umap_table_rows.append({
                            "layer": layer,
                            "n_neighbors": int(nn_),
                            "min_dist": float(md),
                            "figure": wandb.Image(plt.gcf()),
                            "file": out_path,
                        })
                    plt.close()
            # Log UMAP table once
            if self.wandb_run is not None and umap_table_rows:
                cols = ["layer", "n_neighbors", "min_dist", "figure", "file"]
                table = wandb.Table(columns=cols)
                for r in umap_table_rows:
                    table.add_data(r["layer"], r["n_neighbors"], r["min_dist"], r["figure"], r["file"])
                self.wandb_run.log({"umap/table": table})

        if self.config.get("forward_to_csv_simple"):
            print("forward_to_csv_simple")
            eval_tf = get_transforms(self.config["model_config"], split="eval")
            debug = bool(self.config.get("debug"))
            meta_dir = self.config.get("meta_data_dir")
            if not meta_dir:
                raise ValueError(f"meta_dir missing in config {self.config}")
            gene_csv = self.config.get("gene_data_filename")
            if not gene_csv:
                raise ValueError(f"gene_data_filename missing in config {self.config}")
            dataset = get_dataset_from_config(
                dataset_name=self.config["dataset"],
                genes=self.config["model_config"]["genes"],
                split="test",
                debug=debug,
                transforms=eval_tf,
                samples=self.config.get("test_samples"),
                max_len=100 if debug else None,
                only_inputs=False,
                meta_data_dir=meta_dir,
                gene_data_filename=gene_csv,
                return_patient_and_tilepath=True
            )

            print("dataset", len(dataset))

            rows = []
            device = auto_device(self.model)
            self.model.eval()
            for img, y, patient, tile_name in dataset:
                row = {}


                img = img.unsqueeze(0).to(device)
                y_hat = self.model(img)
                for gene in self.config["model_config"]["genes"]:
                    gene_idx = self.model.gene_to_idx[gene]
                    gene_out = y_hat[0, gene_idx].item()

                    row[gene + "_pred"] = gene_out
                    row[gene + "_label"] = float(y[gene_idx])
                    row["patient"] = patient
                    row["tile"] = tile_name
                    print(row)
                    exit(0)
                    rows.append(row)

            if debug:
                exit(0)


        if self.config.get("forward_to_csv"):
            patients = list(self.config.get("test_samples", []))
            genes = self.config["model_config"]["genes"]
            loss_switch = str((self.config.get("model_config") or {}).get("loss_fn_switch", "")).strip().lower()
            use_wmse = loss_switch in {"wmse", "weighted mse"}
            run_name = "wmse" if use_wmse else "wmse_off"
            image_size = int(self.config["model_config"].get("image_size", 224))
            results_dir = os.path.join(self.config["eval_path"], run_name, "predictions")
            results_csv = os.path.join(results_dir, "results.csv")
            device = auto_device(self.model)
            cfg = self.config
            gene_csv = cfg.get("gene_data_filename")
            if not gene_csv:
                raise RuntimeError(f"missing gene_data_filename; dataset={cfg.get('dataset', 'unknown')}")
            project = cfg.get("project") or (cfg.get("model_config") or {}).get("project")
            if not project:
                raise RuntimeError("project missing in config for forward metrics")
            per_patient = bool(self.config.get("forward_metrics_per_patient", False))
            if os.path.exists(results_csv):
                update_forward_metrics_global(
                    self.config["eval_path"], results_csv, project, self.run_name,
                    self.config.get("model_state_path") or self.model_name, bool(self.config.get("debug", False))
                )
                if not bool(self.config.get("debug", False)):
                    append_geneset_predictions_global(
                        self.config["eval_path"], results_csv, self.config, self.run_name, epsilon=0.1
                    )
            else:
                for p in patients:
                    print(f"[Eval] Starting forward_to_csv for patient={p} -> {results_dir}")
                    _ = generate_results(
                        model=self.model,
                        device=device,
                        data_dir=self.config["data_dir"],
                        run_name=run_name,
                        out_path=self.config["eval_path"],
                        patient=p,
                        genes=genes,
                        meta_data_dir=self.config["meta_data_dir"],
                        gene_data_filename=gene_csv,
                        image_size=image_size,
                        max_len=self.config.get("forward_max_tiles") if bool(self.config.get("debug", False)) else None,
                        forward_batch_size=int(self.config.get("forward_batch_size", 32)),
                        forward_num_workers=int(self.config.get("forward_num_workers", 0)),
                    )
                    if per_patient:
                        update_forward_metrics_global(
                            self.config["eval_path"], results_csv, project, self.run_name,
                            self.config.get("model_state_path") or self.model_name, bool(self.config.get("debug", False))
                        )
                        if not bool(self.config.get("debug", False)):
                            append_geneset_predictions_global(
                                self.config["eval_path"], results_csv, self.config, self.run_name, epsilon=0.1
                            )
                if not per_patient:
                    update_forward_metrics_global(
                        self.config["eval_path"], results_csv, project, self.run_name,
                        self.config.get("model_state_path") or self.model_name, bool(self.config.get("debug", False))
                    )
                    if not bool(self.config.get("debug", False)):
                        append_geneset_predictions_global(
                            self.config["eval_path"], results_csv, self.config, self.run_name, epsilon=0.1
                        )
            # Write per-run metrics summary and update aggregated forward_metrics.csv (non-debug)
            if not os.path.exists(results_csv):
                raise RuntimeError(f"results.csv not found at {results_csv}")
            metrics, counts = _compute_metrics_for_csv(results_csv)
            pearson_vals = {k[len("pearson_"):]: v for k, v in metrics.items() if k.startswith("pearson_")}
            mse_vals = {k[len("mse_"):]: v for k, v in metrics.items() if k.startswith("mse_")}
            row = {**metrics}
            row.update({f"count_{g}": counts[g] for g in counts})
            row["pearson_mean"] = _weighted_mean(pearson_vals, counts)
            row["mse_mean"] = _weighted_mean(mse_vals, counts)
            row["wmse"] = bool(use_wmse)
            mc = self.config.get("model_config") or {}
            freeze_encoder_cfg = bool(mc.get("freeze_encoder", False))
            efl = int(mc.get("encoder_finetune_layers", 0) or 0)
            row["freeze_encoder"] = bool(freeze_encoder_cfg)
            row["encoder_finetune_layers"] = int(efl)
            os.makedirs(results_dir, exist_ok=True)
            pd.DataFrame([row]).to_csv(os.path.join(results_dir, "metrics_summary.csv"), index=False)
            if not bool(self.config.get("debug", False)):
                gather_forward_metrics(self.config["eval_path"])
        if self.config.get("lxt"):
            # Delegate to LXT plotting and save under out_path/<model_name>/lxt
            cfg_lxt = dict(self.config)
            lxt_dir = os.path.join(self.config["eval_path"], self.model_name, "lxt")
            os.makedirs(lxt_dir, exist_ok=True)
            cfg_lxt["out_path"] = lxt_dir
            # Provide safe default gamma values for LXT if none supplied
            if not any(k in cfg_lxt for k in ("lxt_conv_gamma", "lxt_lin_gamma", "lxt_conv_gamma_list", "lxt_lin_gamma_list")):
                cfg_lxt["lxt_conv_gamma"] = 0.25
                cfg_lxt["lxt_lin_gamma"] = 0.25
            print(f"[Eval] Starting LXT plots -> {lxt_dir}")
            plot_lxt(self.model, cfg_lxt, run=self.wandb_run, out_path=lxt_dir)
