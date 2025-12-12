import json
import numpy as np
from script.configs.dataset_config import get_dataset_cfg
from script.evaluation.crp_plotting import plot_crp_zennit, _get_composite_and_highest_layer
from script.evaluation.pcx_plotting import plot_pcx
from script.model.lit_model import load_lit_regressor
from script.data_processing.data_loader import get_dataset_from_config
from script.data_processing.image_transforms import get_transforms
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

from typing import Dict, List, Union, Any, Tuple, Iterable
from PIL import Image
import torch
from torchvision.transforms.functional import gaussian_blur
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import zennit.image as zimage
from crp.helper import max_norm

class EvalPipeline:
    def __init__(self, config, run):
        self.config = prepare_cfg(config)
        self.wandb_run = run
        self.model_src = self.config.get("model_config_path")
        self.model = self._load_model()
        # Derive a model-based folder name and ensure base exists
        self.model_name = self._derive_model_name()

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

    # for easy unpacking, dataset needs to return a sample and an idex
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

    def _run_crp_rank(self, ds, layer_name: str, target_index: int, k = 10):
        base = os.path.join(self.config.get("eval_path"), self.model_name)
        os.makedirs(base, exist_ok=True)
        out_fn = os.path.join(base, "crp_rank.json")
        """
        if os.path.exists(out_fn):
            with open(out_fn) as f:
                components = json.load(f)["components"]
            self.config.setdefault("crp_components", components)
            return components
        """
        composite, highest_layer = _get_composite_and_highest_layer(self.model.encoder, self.config["model_config"]["encoder_type"])

        ds = self._RankDataset(ds, target_index)
        fv = FeatureVisualization(CondAttribution(self.model), ds, {layer_name: ChannelConcept()})
        fv.run(composite, 0, len(ds))

        # d_c_sorted: (S, C) — dataset indices of top‑S samples per channel
        # rel_c_sorted: (S, C) — corresponding (normalized) relevance per sample & channel
        # rf_c_sorted: (S, C) — neuron index (RF position) for each selected sample & channel
        # S = Maximization.SAMPLE_SIZE (default 40)
        d_c_sorted, rel_c_sorted, rf_c_sorted = load_maximization(fv.RelMax.PATH, layer_name)

        print("hello")
        #for x in rel_c_sorted:
        #    print(x)
        #print(ds.base.get_tilename(0))
        #for i in range(len(ds)):
        #    print(ds.base[i][1])
        
        n_channels = rel_c_sorted.shape[1]
        k = min(n_channels, k)
        sample_0 = rel_c_sorted[1]
        sample_0 = np.argsort(np.abs(sample_0))
        print(sample_0)
        components = sample_0[:k].tolist()
        
        with open(out_fn, "w") as handle:
            json.dump({"layer": layer_name, "components": components}, handle)
        if not self.config.get("crp_components") and components:
            self.config["crp_components"] = components
        return components

    def plot_grid(ref_c: Dict[int, Any], cmap_dim=1, cmap="bwr", vmin=None, vmax=None, symmetric=True, resize=None, padding=True, figsize=(6, 6)):
        """
        Plots dictionary of reference images as they are returned of the 'get_max_reference' method. To every element in the list crp.imgify is applied with its respective argument values.
    
        Parameters:
        ----------
        ref_c: dict with keys: integer and value: several lists filled with torch.Tensor, np.ndarray or PIL Image
            To every element in the list crp.imgify is applied.
        resize: None or int
            If None, no resizing is applied. If int, sets the maximal aspect ratio of the image.
        padding: boolean
            If True, pads the image into a square shape by setting the alpha channel to zero outside the image.
        figsize: tuple or None
            Size of plt.figure
        cmap_dim: int, 0 or 1 
            Applies the remaining parameters to the first or second element of the tuple list, i.e. plot as heatmap
    
        REMAINING PARAMETERS: correspond to zennit.imgify
    
        Returns:
        --------
        shows matplotlib.pyplot plot
        """
    
        keys = list(ref_c.keys())
        nrows = len(keys)
        value = next(iter(ref_c.values()))
    
        if cmap_dim > 2 or cmap_dim < 1 or cmap_dim == None:
            raise ValueError("'cmap_dim' must be 0 or 1 or None.")
    
        if isinstance(value, Tuple) and isinstance(value[0], Iterable):
            nsubrows = len(value)
            ncols = len(value[0])
        elif isinstance(value, Iterable):
            nsubrows = 1
            ncols = len(value)
        else:
            raise ValueError("'ref_c' dictionary must contain an iterable of torch.Tensor, np.ndarray or PIL Image or a tuple of thereof.")
    
        fig = plt.figure(figsize=figsize)
        outer = gridspec.GridSpec(nrows, 1, wspace=0, hspace=0.2)
    
        for i in range(nrows):
            inner = gridspec.GridSpecFromSubplotSpec(nsubrows, ncols, subplot_spec=outer[i], wspace=0, hspace=0.1)
    
            for sr in range(nsubrows):
    
                if nsubrows > 1:
                    img_list = ref_c[keys[i]][sr]
                else:
                    img_list = ref_c[keys[i]]
                
                for c in range(ncols):
                    ax = plt.Subplot(fig, inner[sr, c])
    
                    if sr == cmap_dim:
                        img = imgify(img_list[c], cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric, resize=resize, padding=padding)
                    else:
                        img = imgify(img_list[c], resize=resize, padding=padding)
    
                    ax.imshow(img)
                    ax.set_xticks([])
                    ax.set_yticks([])
    
                    if sr == 0 and c == 0:
                        ax.set_ylabel(keys[i])
    
                    fig.add_subplot(ax)
                    
        outer.tight_layout(fig)
        outpath = "../evaluation/crp_out/"
        os.mkdirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, self.config["model_state_path"],"crp_out.png")
        fig.savefig(out_path)

    
    def run(self):
        if self.config.get("rank_crp"):
            print("[EvalPipeline] rank_crp case started")
            eval_tf = get_transforms(self.config["model_config"], split="eval")
            if hasattr(self.model, "encoder_is_fully_frozen"):
                self.model.encoder_is_fully_frozen = False

            ds_config = get_dataset_cfg(self.config)
            self.config.update(ds_config)

            gene = self.config.get("gene")
            ds = get_dataset_from_config(
                dataset_name=self.config["model_config"]["dataset"],
                genes=[gene],
                split="test",
                debug=bool(self.config.get("debug")),
                transforms=eval_tf,
                samples=["MISC33"],
                only_inputs=False,
                meta_data_dir=self.config["meta_data_dir"],
                gene_data_filename=self.config["gene_data_filename"],
                return_patient_and_tilepath=True,
                max_len=20 if self.config.get("debug") else None
            )

            crp_str = "crp_demo" if self.config["debug"] else "crp"
            # :10 cuts ../models/
            out_path = os.path.join("../evaluation/", crp_str, self.config["model_config"]["out_path"][10:])
            os.makedirs(out_path, exist_ok=True)
            # Resolve target layer from encoder_type mapping; users no longer override.
            composite, highest_layer = _get_composite_and_highest_layer(self.model.encoder, self.config["model_config"]["encoder_type"])
            resolved_layer = f"encoder.{highest_layer}"
            names = {n for n, _ in self.model.named_modules()}
            if resolved_layer not in names:
                raise KeyError(f"target_layer '{resolved_layer}' not found in model named_modules().")

            ds_len = len(ds)
            print(f"[CRP] Starting CRP (layer='{resolved_layer}') on {ds_len} items -> {out_path}")
            idx = self.model.genes.index(gene)
            self._run_crp_rank(ds, resolved_layer, idx)
        
        if self.config.get("crp"):
            print("[EvalPipeline] crp case started")
            eval_tf = get_transforms(self.config["model_config"], split="eval")
            if hasattr(self.model, "encoder_is_fully_frozen"):
                self.model.encoder_is_fully_frozen = False

            ds_config = get_dataset_cfg(self.config)
            self.config.update(ds_config)

            gene = self.config.get("gene")
            ds = get_dataset_from_config(
                dataset_name=self.config["model_config"]["dataset"],
                genes=[gene],
                split="test",
                debug=bool(self.config.get("debug")),
                transforms=eval_tf,
                samples=self.config.get("patients"),
                only_inputs=False,
                meta_data_dir=self.config["meta_data_dir"],
                gene_data_filename=self.config["gene_data_filename"],
                return_patient_and_tilepath=True,
                max_len=5 if self.config.get("debug") else None
            )

            crp_str = "crp_demo" if self.config["debug"] else "crp"
            # :10 cuts ../models/
            out_path = os.path.join("../evaluation/", crp_str, self.config["model_config"]["out_path"][10:])
            os.makedirs(out_path, exist_ok=True)
            # Resolve target layer from encoder_type mapping; users no longer override.
            composite, layername = _get_composite_and_highest_layer(self.model.encoder, self.config["model_config"]["encoder_type"])
            resolved_layer = f"encoder.{layername}"
            names = {n for n, _ in self.model.named_modules()}
            if resolved_layer not in names:
                raise KeyError(f"target_layer '{resolved_layer}' not found in model named_modules().")

            ds_len = len(ds)
            print(f"[CRP] Starting CRP (layer='{resolved_layer}') on {ds_len} items -> {out_path}")

            idx = self.model.genes.index(gene)
            channels = self.config.get("crp_components")
            if not channels:
                channels = self._run_crp_rank(ds, resolved_layer, idx)



            from crp.concepts import ChannelConcept
            from crp.image import plot_grid

            crp_ds = self._RankDataset(ds, idx)
            attribution = CondAttribution(self.model)
            cc = ChannelConcept()
            layer_map = {layer : cc for layer in [resolved_layer]}
            fv = FeatureVisualization(attribution, crp_ds, layer_map, preprocess_fn=eval_tf, path="../evaluation/crp_demo")
            saved_files = fv.run(composite, 0, len(crp_ds), 32, 100)
            ref_c = fv.get_max_reference(channels, resolved_layer, "relevance", (0, 8), composite=composite, plot_fn=None)
            res = self.plot_grid(ref_c, figsize=(6, 9), cmap = "coldnhot")
            from crp.image import vis_opaque_img
            ref_c = fv.get_max_reference(channels, resolved_layer, "relevance", (0, 8), rf=True, composite=composite, plot_fn=vis_opaque_img)

            self.plot_grid(ref_c, figsize=(6, 5), padding=False)
            return
            
            for c in channels:
                plot_crp_zennit(
                    self.model,
                    dataset=ds,
                    model_config=self.config["model_config"],
                    out_path=out_path,
                    target_layer_name=resolved_layer,
                    composite=composite,
                    component=None,
                    target_index=idx,
                    run=self.wandb_run,
                )


        if self.config.get("pcx"):
            # Enforce: always use model genes; eval config must not set 'genes'.
            model_config = self.config.get("model_config")
            # Route outputs under eval_path/<model_name>/pcx
            out_path = "../evaluation"
            if self.config.get("debug"):
                out_path = os.path.join(out_path, "debug")
            out_path = os.path.join(out_path, "pcx")
            out_path = os.path.join(out_path, model_config["project"])
            self.config["out_path"] = out_path

            os.makedirs(out_path, exist_ok=True)
            # Ensure genes come from model
            gene = self.config.get("gene")
            self.config["genes"] = [gene]
            print(f"[CRP] Starting PCX -> {self.config['out_path']}")
            plot_pcx(self.model, self.config, run=self.wandb_run, out_path=self.config["out_path"])
