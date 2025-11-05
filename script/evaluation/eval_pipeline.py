import os
import hashlib
from typing import Any, Dict, Optional
import numpy as np
import torch
from umap import UMAP
import matplotlib.pyplot as plt
import wandb

#from script.evaluation.crp_plotting import plot_crp, plot_crp_zennit, plot_crp2
from script.evaluation.lrp_plotting import plot_lrp, plot_lrp_custom
from script.evaluation.lxt_plotting import plot_lxt
from script.evaluation.scatter_plotting import plot_scatter
from script.evaluation.generate_results import generate_results
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

class EvalPipeline:
    def __init__(self, config, run):
        # Avoid double-prep: eval_main prepares cfg (merges dataset cfg, sets eval_path, writes config)
        self.config = dict(config)
        self.wandb_run = run
        # Single source of truth for run_name: require it in config
        self.run_name = self.config.get("run_name")
        if not self.run_name:
            raise ValueError("Config must provide 'run_name' (single source of truth).")
        self.model_src = self.config.get("model_config_path")
        self.model = self._load_model()
        # Force evaluation on an accelerator; require GPU availability
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
        # Set genes from model config
        self.config["genes"] = list(mc_genes)
        # Defaults for single-gene cases
        if not self.config.get("gene") and self.config.get("genes"):
            self.config["gene"] = str(self.config["genes"][0])
        # Provide a default patient (first test sample) for cases that need one
        if not self.config.get("patient"):
            test_samples = self.config.get("test_samples") or model_cfg.get("test_samples")
            if isinstance(test_samples, (list, tuple)) and len(test_samples) > 0:
                self.config["patient"] = str(test_samples[0])

    def _load_model(self):
        state_dicts = collect_state_dicts(self.config)
        return load_lit_regressor(self.config["model_config"], state_dicts)

    def _derive_model_name(self) -> str:
        def _rel_model_path(p: str) -> str:
            s = str(p).replace("\\", "/")
            if "/models/" in s:
                s = s.split("/models/", 1)[1]
            elif s.startswith("../models/"):
                s = s[len("../models/"):]
            return s.strip("/")

        parts = []
        ms = self.config.get("model_state_path")
        enc = self.config.get("encoder_state_path")
        head = self.config.get("gene_head_state_path")
        if ms:
            parts.append(_rel_model_path(ms))
        else:
            if enc:
                parts.append(_rel_model_path(enc))
            if head:
                parts.append(_rel_model_path(head))

        name = os.path.join(*parts) if parts else str(self.run_name)
        MAX_LEN = 160
        if len(name) <= MAX_LEN:
            return name
        tokens = [t for t in name.split("/") if t]
        tail = "/".join(tokens[-2:]) if len(tokens) >= 2 else tokens[-1]
        h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        short = f"{tail}/__{h}"
        return short[:MAX_LEN]

    def run(self):
        if self.config.get("lrp"):
            lrp_backend = str(self.config.get("lrp_backend", "zennit")).lower()
            eval_tf = get_transforms(self.config["model_config"], split="eval")
            debug = bool(self.config.get("debug", False))
            # Directly resolve metadata CSV using eval override with model_config fallback
            cfg = self.config
            meta_dir = cfg.get("meta_data_dir") or cfg["model_config"].get("meta_data_dir", "/meta_data/")
            gene_csv = cfg.get("gene_data_filename") or cfg.get("model_config", {}).get("gene_data_filename", "gene_data.csv")
            ds = get_dataset_from_config(
                dataset_name=self.config["model_config"]["dataset"],
                genes=None,
                split="test",
                debug=debug,
                transforms=eval_tf,
                samples=self.config.get("test_samples"),
                max_len=self.config.get("max_len") if debug else None,
                only_inputs=False,
                meta_data_dir=meta_dir,
                gene_data_filename=gene_csv,
            )
            n = min(int(self.config.get("lrp_max_items", 10)), len(ds))
            loader = DataLoader(Subset(ds, list(range(n))), batch_size=1, shuffle=False)
            # Prepare a local save dir for LRP outputs regardless of W&B.
            lrp_dir = os.path.join(self.config["eval_path"], self.model_name, "lrp")
            os.makedirs(lrp_dir, exist_ok=True)
            print(f"[Eval] Starting LRP (backend={lrp_backend}) on {n} items -> {lrp_dir}")
            if lrp_backend == "custom":
                plot_lrp_custom(self.model, loader, run=self.wandb_run, save_dir=lrp_dir)
            else:
                plot_lrp(self.model, loader, run=self.wandb_run, save_dir=lrp_dir)

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
            cfg_scatter["out_path"] = os.path.join(self.config["eval_path"], self.model_name, "scatter")
            print(f"[Eval] Starting scatter plots -> {cfg_scatter['out_path']}")
            plot_scatter(
                cfg_scatter,
                self.model,
                wandb_run=self.wandb_run
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
            # Directly resolve metadata CSV using eval override with model_config fallback
            cfg = self.config
            meta_dir = cfg.get("meta_data_dir") or cfg["model_config"].get("meta_data_dir", "/meta_data/")
            gene_csv = cfg.get("gene_data_filename") or cfg.get("model_config", {}).get("gene_data_filename", "gene_data.csv")
            ds = get_dataset_from_config(
                dataset_name=self.config["model_config"]["dataset"],
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

        if self.config.get("forward_to_csv"):
            # Use explicit test_samples; dataset config populates this when absent.
            patients = list(self.config.get("test_samples", []))

            genes = self.config["model_config"]["genes"]
            run_name = self.model_name
            image_size = int(self.config["model_config"].get("image_size", 224))

            # Early‑exit if the aggregate target already exists to avoid doing any forwards
            results_dir = os.path.join(self.config["eval_path"], run_name, "predictions")
            results_csv = os.path.join(results_dir, "results.csv")
            if os.path.exists(results_csv):
                print(f"[EvalPipeline] predictions already exist at {results_csv}; skipping forward_to_csv.")
            else:
                device = auto_device(self.model)
                # Directly resolve metadata CSV using eval override with model_config fallback
                cfg = self.config
                gene_csv = cfg.get("gene_data_filename") or cfg.get("model_config", {}).get("gene_data_filename", "gene_data.csv")
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
        if self.config.get("lxt"):
            # Delegate to LXT plotting and save under out_path/<model_name>/lxt
            cfg_lxt = dict(self.config)
            lxt_dir = os.path.join(self.config["eval_path"], self.model_name, "lxt")
            os.makedirs(lxt_dir, exist_ok=True)
            cfg_lxt["out_path"] = lxt_dir
            print(f"[Eval] Starting LXT plots -> {lxt_dir}")
            plot_lxt(self.model, cfg_lxt, run=self.wandb_run)
