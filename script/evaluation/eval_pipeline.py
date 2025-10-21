import os
import hashlib
from typing import Any, Dict, Optional
import numpy as np
import torch
from umap import UMAP
import matplotlib.pyplot as plt

#from script.evaluation.crp_plotting import plot_crp, plot_crp_zennit, plot_crp2
from script.evaluation.lrp_plotting import plot_lrp, plot_lrp_custom
from script.evaluation.lxt_plotting import plot_lxt
from script.evaluation.pcx_plotting import plot_pcx
from script.evaluation.tri_plotting import plot_triptych_from_merge
from script.evaluation.scatter_plotting import plot_scatter
from script.evaluation.generate_results import generate_results
from script.model.model_factory import get_encoder
from script.model.lit_model import load_lit_regressor
from script.data_processing.data_loader import get_dataset_from_config, get_dataset
from script.data_processing.image_transforms import get_transforms, get_eval_transforms
from torch.utils.data import DataLoader, Subset
from script.main_utils import prepare_cfg
from script.evaluation.eval_helpers import (
    auto_device,
    collect_state_dicts,
)
from script.train.lit_train_sae import SAETrainerPipeline

class EvalPipeline:
    def __init__(self, config, run):
        self.config = prepare_cfg(config)
        self.wandb_run = run
        # Single source of truth for run_name: require it in config
        self.run_name = self.config.get("run_name")
        if not self.run_name:
            raise ValueError("Config must provide 'run_name' (single source of truth).")
        self.model_src = self.config.get("model_config_path")
        self.model = self._load_model()
        # Derive a folder name from provided state paths (preferred over run_name)
        self.model_name = self._derive_model_name()
        os.makedirs(os.path.join(self.config["out_path"], self.model_name), exist_ok=True)

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
            ds = get_dataset_from_config(
                dataset_name=self.config["model_config"]["dataset"],
                genes=None,
                split="val",
                debug=debug,
                transforms=eval_tf,
                samples=None,
                only_inputs=True,
                meta_data_dir=self.config["model_config"]["meta_data_dir"],
                gene_data_filename=self.config["model_config"]["gene_data_filename"]
            )
            n = min(10, len(ds))
            loader = DataLoader(Subset(ds, list(range(n))), batch_size=1, shuffle=False)
            if lrp_backend == "custom":
                plot_lrp_custom(self.model, loader, run=self.wandb_run)
            else:
                plot_lrp(self.model, loader, run=self.wandb_run)

        if self.config.get("pcx"):
            pcx_backend = str(self.config.get("pcx_backend", "zennit")).lower()
            # For now, PCX uses the zennit/CRP-based pipeline in cluster_functions.
            # The backend flag is accepted for parity and future extension.
            cfg_pcx = dict(self.config)
            cfg_pcx["out_path"] = os.path.join(self.config["out_path"], self.model_name, "pcx")
            plot_pcx(self.model, cfg_pcx, run=self.wandb_run)
        if self.config.get("diff"):
            plot_triptych_from_merge(
                self.config["data_dir"],
                self.config["patient"],
                self.config["gene"],
                os.path.join(self.config["out_path"], self.model_name, "diff"),
                is_online=bool(self.wandb_run),
                wandb_run=self.wandb_run,
            )

        if self.config.get("scatter"):
            cfg_scatter = dict(self.config)
            cfg_scatter["out_path"] = os.path.join(self.config["out_path"], self.model_name, "scatter")
            plot_scatter(
                cfg_scatter,
                self.model,
                wandb_run=self.wandb_run
            )

        if self.config.get("sae"):
            # Train SAE as an evaluation case, mirroring other case patterns.
            # Route outputs under out_path/run_name/train_sae and avoid nested W&B runs.
            cfg_sae = dict(self.config)
            # Ensure encoder_type is available to the SAE trainer; fallback to model_config.
            if "encoder_type" not in cfg_sae and isinstance(self.config.get("model_config"), dict):
                enc_type = self.config["model_config"].get("encoder_type")
                if enc_type:
                    cfg_sae["encoder_type"] = enc_type
            # Use only inputs in datamodule and prevent nested W&B init inside the trainer.
            cfg_sae["train_sae"] = True
            cfg_sae["log_to_wandb"] = False
            # Case-specific output directory for SAE; follow case pattern (use out_path)
            sae_dir = os.path.join(self.config["out_path"], self.model_name, "sae")
            os.makedirs(sae_dir, exist_ok=True)
            cfg_sae["out_path"] = sae_dir
            cfg_sae["model_dir"] = sae_dir  # keep checkpoints enabled
            # Pass the already-loaded encoder; do not wire post-train attachment here
            SAETrainerPipeline(cfg_sae, run=self.wandb_run, encoder=self.model.encoder).run()

        if self.config.get("umap"):
            # Generate UMAP visualizations from a specified model layer
            layer = self.config.get("umap_layer")
            if not layer or not isinstance(layer, str):
                raise ValueError("'umap' is enabled, but 'umap_layer' is missing. Provide a dot-path layer name.")

            umap_dir = os.path.join(self.config["out_path"], self.model_name, "umap")
            os.makedirs(umap_dir, exist_ok=True)

            # Build eval loader (inputs only)
            eval_tf = get_transforms(self.config["model_config"], split="eval")
            debug = bool(self.config.get("debug", False))
            ds = get_dataset_from_config(
                dataset_name=self.config["model_config"]["dataset"],
                genes=None,
                split="val",
                debug=debug,
                transforms=eval_tf,
                samples=None,
                only_inputs=True,
                meta_data_dir=self.config["model_config"]["meta_data_dir"],
                gene_data_filename=self.config["model_config"]["gene_data_filename"]
            )
            bs = int(self.config.get("umap_batch_size", 32))
            loader = DataLoader(ds, batch_size=bs, shuffle=False)

            device = auto_device(self.model)
            self.model.eval()

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

            for nn_ in n_neighbors_list:
                for md in min_dist_list:
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
                    plt.savefig(os.path.join(umap_dir, fn), dpi=200)
                    plt.close()

        if self.config.get("forward_to_csv"):
            patients = sorted([
                f.name for f in os.scandir(self.config["data_dir"]) if f.is_dir() and not f.name.startswith((".", "_"))
            ])

            genes = self.config["model_config"]["genes"]
            run_name = self.model_name
            image_size = int(self.config["model_config"].get("image_size", 224))

            # Early‑exit if the aggregate target already exists to avoid doing any forwards
            results_dir = os.path.join(self.config["out_path"], run_name, "predictions")
            results_csv = os.path.join(results_dir, "results.csv")
            if os.path.exists(results_csv):
                print(f"[EvalPipeline] predictions already exist at {results_csv}; skipping forward_to_csv.")
            else:
                device = auto_device(self.model)
                for p in patients:
                    _ = generate_results(
                        model=self.model,
                        device=device,
                        data_dir=self.config["data_dir"],
                        run_name=run_name,
                        out_path=self.config["out_path"],
                        patient=p,
                        genes=genes,
                        meta_data_dir=self.config["meta_data_dir"],
                        gene_data_filename=self.config["gene_data_filename"],
                        image_size=image_size,
                    )
        if self.config.get("lxt"):
            # Delegate to dedicated LXT plotting function
            plot_lxt(self.model, self.config, run=self.wandb_run)
