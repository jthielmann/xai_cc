import itertools
import os
import json
import torch
import numpy as np
from typing import Any, Dict
from script.configs.dataset_config import get_dataset_cfg
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
from crp.helper import load_maximization
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

    # for easy unpacking
    class _RankDataset:
        def __init__(self, base, target_idx: int):
            self.base = base
            self.target_idx = int(target_idx)
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            sample = self.base[idx]
            x = sample[0]
            return x, self.target_idx

    def _run_crp_rank(self, ds, layer_name: str, target_index: int):
        k = int(self.config.get("crp_rank_k"))
        if k <= 0:
            raise ValueError(f"crp_rank_k must be greater than zero, got {k}.")
        composite, _ = _get_composite_and_layer(
            self.model.encoder, self.config.get("encoder_type")
        )

        ds = self._RankDataset(ds, target_index)
        fv = FeatureVisualization(CondAttribution(self.model), ds, {layer_name: ChannelConcept()})
        fv.run(composite, 0, len(ds))
        _, rel_c_sorted, _ = load_maximization(fv.RelMax.PATH, layer_name)
        n_channels = rel_c_sorted.shape[1]
        if k > n_channels:
            raise ValueError(
                f"crp_rank_k={k} exceeds channel count {n_channels} for {layer_name}."
            )
        best = rel_c_sorted[0]
        topk = np.argsort(-best)[:k].tolist()
        refs = fv.get_max_reference(concept_ids=topk, layer_name=layer_name)
        components = list(refs.keys())
        base = os.path.join(self.config.get("eval_path"), self.model_name)
        os.makedirs(base, exist_ok=False)
        out_fn = os.path.join(base, "crp_rank.json")
        with open(out_fn, "w") as handle:
            json.dump({"layer": layer_name, "components": components}, handle)
        if not self.config.get("crp_components") and components:
            self.config["crp_components"] = components
        return components

    def run(self):
        if self.config.get("crp"):
            genes_model = []
            for g in self.config["model_config"]["genes"]:
                if g.lower().startswith("unnamed") or g.lower() in {"x", "y"}:
                    continue
                genes_model.append(g)

            eval_tf = get_transforms(self.config["model_config"], split="eval")
            genes_target = self.config["genes"]
            if hasattr(self.model, "encoder_is_fully_frozen"):
                self.model.encoder_is_fully_frozen = False

            ds_config = get_dataset_cfg(self.config)
            self.config.update(ds_config)


            ds = get_dataset_from_config(
                dataset_name=self.config["model_config"]["dataset"],
                genes=genes_target,
                split="test",
                debug=bool(self.config.get("debug")),
                transforms=eval_tf,
                samples=self.config.get("test_samples"),
                only_inputs=False,
                meta_data_dir=self.config["meta_data_dir"],
                gene_data_filename=self.config["gene_data_filename"],
                return_patient_and_tilepath=True,
            )


            items = 100 if self.config["debug"] else len(ds)
            ds = Subset(ds, list(range(items)))
            crp_str = "crp_demo" if self.config["debug"] else "crp"
            # :10 cuts ../models/
            out_path = os.path.join("../evaluation/", crp_str, self.config["model_config"]["out_path"][10:])
            os.makedirs(out_path, exist_ok=True)
            # Resolve target layer from encoder_type mapping; users no longer override.
            composite, layername = _get_composite_and_layer(self.config.get("encoder_type"))
            resolved_layer = f"encoder.{layername}"
            names = {n for n, _ in self.model.named_modules()}
            if resolved_layer not in names:
                raise KeyError(f"target_layer '{resolved_layer}' not found in model named_modules().")

            ds_len = len(ds)
            print(f"[CRP] Starting CRP (layer='{resolved_layer}') on {ds_len} items -> {out_path}")

            for g in genes_target:
                idx = self.model.genes.index(g)
                top_channels = self._run_crp_rank(ds, resolved_layer, idx)

                for c in top_channels:

                    plot_crp_zennit(
                        self.model,
                        ds,
                        run=self.wandb_run,
                        max_items=items,
                        out_path=out_path,
                        layer_name=resolved_layer,
                        components=self.config.get("crp_components"),
                        target_index=idx,
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
