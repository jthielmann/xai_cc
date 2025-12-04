import itertools
import os
import json
import torch
from typing import Any, Dict

from script.evaluation.crp_plotting import plot_crp_zennit, plot_crp, _get_composite_and_layer
from script.evaluation.pcx_plotting import plot_pcx
from script.model.lit_model import load_lit_regressor
from script.data_processing.data_loader import get_dataset_from_config
from script.data_processing.image_transforms import get_transforms
from torch.utils.data import DataLoader, Subset
from script.main_utils import prepare_cfg
import hashlib
import os
from script.evaluation.eval_helpers import (
    collect_state_dicts
)
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.visualization import FeatureVisualization

class EvalPipeline:
    def __init__(self, config, run):
        self.config = prepare_cfg(config)
        self.wandb_run = run
        self.model_src = self.config.get("model_config_path")
        self.model = self._load_model()
        # Derive a model-based folder name and ensure base exists
        self.model_name = self._derive_model_name()
        base = os.path.join(self.config.get("eval_path", self.config.get("out_path", "./xai_out")), self.model_name)
        os.makedirs(base, exist_ok=True)

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

        name = os.path.join(*parts) if parts else "eval"
        MAX_LEN = 160
        if len(name) <= MAX_LEN:
            return name
        tokens = [t for t in name.split("/") if t]
        tail = "/".join(tokens[-2:]) if len(tokens) >= 2 else tokens[-1]
        h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        short = f"{tail}/__{h}"
        return short[:MAX_LEN]

    def _run_crp_rank(self, ds, layer_name: str, max_items: int):
        enc = getattr(self.model, "encoder", self.model)
        composite, _ = _get_composite_and_layer(enc)
        if max_items > 0:
            ds = Subset(ds, list(range(min(max_items, len(ds)))))
        n = len(ds)
        fv = FeatureVisualization(CondAttribution(self.model), ds, {layer_name: ChannelConcept()})
        fv.run(composite, 0, n)
        k = self.config.get("crp_rank_k")
        refs = fv.get_max_reference(layer=layer_name, k=k)
        components = list(refs.keys())
        base = os.path.join(self.config.get("eval_path", self.config.get("out_path", "./xai_out")), self.model_name)
        os.makedirs(base, exist_ok=True)
        out_fn = os.path.join(base, "crp_rank.json")
        with open(out_fn, "w") as handle:
            json.dump({"layer": layer_name, "components": components}, handle)
        if not self.config.get("crp_components") and components:
            self.config["crp_components"] = components

    def run(self):
        if self.config.get("crp"):
            # Ensure gradients flow for attribution even if encoder was frozen during training
            if hasattr(self.model, "encoder_is_fully_frozen"):
                self.model.encoder_is_fully_frozen = False
            crp_backend = str(self.config.get("crp_backend", "zennit")).lower()
            eval_tf = get_transforms(self.config["model_config"], split="eval")
            # Use TEST split from the selected dataset; honor configured test_samples
            # Directly resolve metadata CSV using eval override with model_config fallback
            cfg = self.config
            mc_genes = cfg.get("model_config", {}).get("genes")
            if not mc_genes:
                raise ValueError("Trained model config does not contain 'genes'; cannot run CRP.")
            genes = []
            for g in mc_genes:
                gs = str(g).strip()
                if not gs:
                    continue
                if gs.lower().startswith("unnamed"):
                    continue
                genes.append(gs)
            if not genes:
                raise ValueError("No valid genes after excluding empty/unnamed entries.")
            target_gene = cfg.get("gene")
            user_genes = cfg.get("genes")
            if not target_gene and user_genes is not None:
                if isinstance(user_genes, str):
                    target_gene = user_genes.strip()
                elif isinstance(user_genes, (list, tuple)):
                    if len(user_genes) == 1:
                        target_gene = str(user_genes[0]).strip()
                    else:
                        raise ValueError("CRP expects one gene; set 'gene' or a single-element 'genes'.")
            if not target_gene:
                raise ValueError("CRP requires a target gene via 'gene' or a single-element 'genes'.")
            if target_gene not in genes:
                raise ValueError(f"target gene {target_gene!r} not found in model genes list.")
            target_index = genes.index(target_gene)
            cfg["gene"] = target_gene
            cfg["genes"] = genes
            meta_dir = cfg.get("meta_data_dir") or cfg["model_config"].get("meta_data_dir", "/meta_data/")
            gene_csv = cfg.get("gene_data_filename") or cfg.get("model_config", {}).get("gene_data_filename", "gene_data.csv")
            ds = get_dataset_from_config(
                dataset_name=self.config["model_config"]["dataset"],
                genes=cfg["genes"],
                split="test",
                debug=bool(self.config.get("debug", False)),
                transforms=eval_tf,
                samples=self.config.get("test_samples"),
                only_inputs=False,
                meta_data_dir=meta_dir,
                gene_data_filename=gene_csv,
                return_patient_and_tilepath=bool(cfg.get("crp_rank_plot_sidecar")),
            )
            # Optional truncation via config: crp_max_items; default: use all
            max_items = int(self.config.get("crp_max_items", 0) or 0)
            if max_items > 0:
                ds_subset = Subset(ds, list(range(min(max_items, len(ds)))))
            else:
                ds_subset = ds
            out_path = os.path.join(self.config.get("eval_path", self.config.get("out_path", "./xai_out")), self.model_name, "crp")
            os.makedirs(out_path, exist_ok=True)
            # Require explicit target_layer in config; allow special value 'encoder' which maps to a model-specific default
            target_layer = self.config.get("target_layer")
            if not target_layer or not isinstance(target_layer, str):
                raise ValueError("CRP requires 'target_layer' in config (e.g., 'encoder' or a dot-path layer name).")
            enc = getattr(self.model, "encoder", self.model)
            _, default_layer_name = _get_composite_and_layer(enc)
            if target_layer == "encoder":
                # Fully-qualified name if model has an encoder submodule
                resolved_layer = (
                    f"encoder.{default_layer_name}" if enc is not self.model else default_layer_name
                )
            else:
                resolved_layer = target_layer
            # Validate target_layer exists on model for explicit names
            if target_layer != "encoder":
                names = {n for n, _ in self.model.named_modules()}
                if resolved_layer not in names:
                    raise KeyError(f"target_layer '{resolved_layer}' not found in model named_modules().")

            if bool(self.config.get("crp_rank")):
                self._run_crp_rank(ds_subset, resolved_layer, max_items)

            ds_len = len(ds_subset)
            print(f"[CRP] Starting CRP (backend={crp_backend}, layer='{resolved_layer}') on {ds_len} items -> {out_path}")
            if crp_backend == "custom":
                plot_crp(
                    self.model,
                    ds_subset,
                    run=self.wandb_run,
                    out_path=out_path,
                    layer_name=resolved_layer,
                    target=target_index,
                )
            else:
                sidecar_rows = []
                def _sidecar(**row):
                    sidecar_rows.append(row)
                plot_crp_zennit(
                    self.model,
                    ds_subset,
                    run=self.wandb_run,
                    max_items=max_items if max_items > 0 else None,
                    out_path=out_path,
                    layer_name=resolved_layer,
                    components=self.config.get("crp_components"),
                    target_index=target_index,
                    sidecar_handle=_sidecar if bool(self.config.get("crp_rank_plot_sidecar")) else None,
                )
                if self.config.get("crp_rank_plot_sidecar") and sidecar_rows:
                    meta_fn = os.path.join(out_path, "crp_rank_plot_sidecar.jsonl")
                    with open(meta_fn, "w") as handle:
                        for row in sidecar_rows:
                            handle.write(json.dumps(row) + "\n")

        if self.config.get("pcx"):
            # Enforce: always use model genes; eval config must not set 'genes'.
            model_cfg = self.config.get("model_config")
            if "genes" in self.config and self.config["genes"] is not None:
                raise ValueError(
                    "CRP/PCX config must not set 'genes'. The trained model's genes are always used."
                )
            mc_genes = model_cfg.get("genes")
            if not mc_genes:
                raise ValueError("Trained model config does not contain 'genes'; cannot run PCX.")
            cfg_pcx = dict(self.config)
            # Route outputs under eval_path/<model_name>/pcx
            cfg_pcx["out_path"] = os.path.join(self.config.get("eval_path", self.config.get("out_path", "./xai_out")), self.model_name, "pcx")
            os.makedirs(cfg_pcx["out_path"], exist_ok=True)
            # Ensure genes come from model
            cfg_pcx["genes"] = list(mc_genes)
            print(f"[CRP] Starting PCX -> {cfg_pcx['out_path']}")
            plot_pcx(self.model, cfg_pcx, run=self.wandb_run, out_path=cfg_pcx["out_path"])
