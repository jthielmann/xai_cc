import itertools
import os
import torch
from typing import Any, Dict, Optional

#from script.evaluation.crp_plotting import plot_crp, plot_crp_zennit, plot_crp2
from script.evaluation.lrp_plotting import plot_lrp, plot_lrp_custom
from script.evaluation.pcx_plotting import plot_pcx
from script.evaluation.tri_plotting import plot_triptych_from_merge
from script.evaluation.scatter_plotting import plot_scatter
from script.evaluation.generate_results import generate_results, log_patient_hist_from_csv
from script.model.model_factory import get_encoder
from script.model.lit_model import load_model
from script.data_processing.data_loader import get_dataset_from_config, get_dataset
from script.data_processing.image_transforms import get_transforms, get_eval_transforms
from torch.utils.data import DataLoader, Subset
from script.main_utils import prepare_cfg

class EvalPipeline:
    def __init__(self, config, run):
        self.config = prepare_cfg(config)
        self.wandb_run = run
        self.model_src = self.config.get("model_config_path")
        self.model = self._load_model()

    def _auto_device(self):
        try:
            return next(self.model.parameters()).device
        except Exception:
            return torch.device(
                "cuda" if torch.cuda.is_available() else
                ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            )

    def _load_state_dict_from_path(self, path: str) -> Dict[str, Any]:
        state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError(f"State dict file {path!r} did not contain a dictionary")
        return state

    def _normalize_state_dicts(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            raw = raw["state_dict"]
        if any(k in raw for k in ("encoder", "gene_heads", "sae")):
            return {k: raw[k] for k in ("encoder", "gene_heads", "sae") if k in raw}
        return {"encoder": raw}

    def _collect_state_dicts(self) -> Optional[Dict[str, Any]]:
        if self.config.get("encoder_state_path", None) and self.config.get("model_state_path", None):
            raise RuntimeError(
                f"corrupted config: encoder_state_path \n{self.config.get('encoder_state_path')}"
                f"\n{self.config.get('model_state_path')}"
            )

        if self.config.get("encoder_state_path", None) and self.config.get("gene_head_state_path", None):
            paths = {
                "encoder": self.config.get("encoder_state_path"),
                "gene_heads": self.config.get("gene_head_state_path"),
                "sae": self.config.get("sae_state_path"),
            }
            loaded = {k: self._load_state_dict_from_path("../models" + p) for k, p in paths.items() if p}
            if loaded:
                return loaded
        elif self.config.get("model_state_path") and not self.config.get("encoder_state_path", None):
            path = self.config.get("model_state_path") + "/best_model.pth"
            bundled = self._load_state_dict_from_path(path)
            return self._normalize_state_dicts(bundled)
        else:
            raise RuntimeError(
                f"corrupted config:\n"
                f"encoder_state_path \n{self.config.get('encoder_state_path', 'None')}"
                f"gene_head_state_path \n{self.config.get('gene_head_state_path', 'None')}"
                f"model_state_path \n{self.config.get('model_state_path', 'None')}"
            )

    def _load_model(self):
        state_dicts = self._collect_state_dicts()
        return load_model(self.config["model_config"], state_dicts)

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

        """if self.config.get("crp"):
            crp_backend = str(self.config.get("crp_backend", "zennit")).lower()
            eval_tf = get_transforms(self.config["model_config"], split="eval")
            ds = get_dataset_from_config(
                dataset_name=self.config["dataset"],
                genes=None,
                split="val",
                debug=bool(self.config.get("debug", False)),
                transforms=eval_tf,
                samples=None,
                only_inputs=False,
            )
            n = min(10, len(ds))
            ds_subset = Subset(ds, list(range(n)))
            if crp_backend == "custom":
                plot_crp2(self.model, ds_subset, run=self.wandb_run)
            else:
                plot_crp_zennit(self.model, ds_subset, run=self.wandb_run, max_items=n)"""

        if self.config.get("pcx"):
            pcx_backend = str(self.config.get("pcx_backend", "zennit")).lower()
            # For now, PCX uses the zennit/CRP-based pipeline in cluster_functions.
            # The backend flag is accepted for parity and future extension.
            plot_pcx(self.model, self.config, run=self.wandb_run)
        if self.config.get("diff"):
            plot_triptych_from_merge(
                self.config["data_dir"],
                self.config["patient"],
                self.config["gene"],
                self.config["out_path"],
                is_online=bool(self.wandb_run),
                wandb_run=self.wandb_run,
            )
        if self.config.get("scatter"):
            plot_scatter(
                self.config,
                self.model,
                wandb_run=self.wandb_run
            )
        if self.config.get("forward_to_csv"):
            patients = [f.name for f in os.scandir(self.config["data_dir"]) if f.is_dir() and not str(f).startswith(".") and not str(f).startswith("_")]

            results_dir = self.config.get(
                "results_dir",
                os.path.join(self.config.get("out_path", "."), "forward_results")
            )
            genes = self.config["model_config"]["genes"]

            device = self._auto_device()

            for p in patients:
                csv_path = generate_results(
                    model=self.model,
                    device=device,
                    data_dir=self.config["data_dir"],
                    run_name=self.config["model_config"]["name"],
                    patient=p,
                    genes=genes,
                    meta_data_dir=self.config["meta_data_dir"],
                    gene_data_filename=self.config["gene_data_filename"]
                )
                """
                # Log prediction vs. original histograms to W&B (and save an image)
                log_patient_hist_from_csv(
                    results_csv=csv_path,
                    data_dir=self.config["data_dir"],
                    patient=p,
                    genes=genes,
                    results_dir=results_dir
                )
                """
        if self.config.get("lxt"):
            import torch
            import itertools
            from PIL import Image
            from torchvision.models import vision_transformer

            from zennit.image import imgify
            from zennit.composites import LayerMapComposite
            import zennit.rules as z_rules

            from lxt.efficient import monkey_patch, monkey_patch_zennit

            monkey_patch(vision_transformer, verbose=True)
            monkey_patch_zennit(verbose=True)
            genes = self.config["model_config"]["genes"]

            eval_tf = get_eval_transforms(image_size=224)
            base_ds = get_dataset(
                data_dir=self.config["data_dir"],
                genes=genes,
                transforms=eval_tf,
                samples=None,
                only_inputs=False,
                meta_data_dir="metadata",
                gene_data_filename="gene_log1p.csv"
            )
            input_tensor, label = base_ds[0]
            from torchvision.utils import save_image
            save_image(input_tensor, f"{str(label[0])}mytile.png")
            input_tensor = input_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            Image.fromarray(input_tensor).save(f"{str(label[0])}mytilePIL.png")

            exit(0)
            input_tensor = input_tensor.unsqueeze(0)
            heatmaps = []
            for conv_gamma, lin_gamma in itertools.product([0.25], [0]):
                input_tensor.grad = None  # Reset gradients
                print("Gamma Conv2d:", conv_gamma, "Gamma Linear:", lin_gamma)

                # Define rules for the Conv2d and Linear layers using 'zennit'
                # LayerMapComposite maps specific layer types to specific LRP rule implementations
                zennit_comp = LayerMapComposite([
                    (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
                    (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
                ])

                # Register the composite rules with the model
                zennit_comp.register(self.model)

                # Forward pass with gradient tracking enabled
                y = self.model(input_tensor.requires_grad_())

                y[0, 0].backward()

                # Remove the registered composite to prevent interference in future iterations
                zennit_comp.remove()

                # Calculate the relevance by computing Input*Gradient
                # This is the final step of LRP to get the pixel-wise explanation
                heatmap = (input_tensor * input_tensor.grad).sum(1)

                # Normalize relevance between [-1, 1] for plotting
                heatmap = heatmap / abs(heatmap).max()

                # Store the normalized heatmap
                heatmaps.append(heatmap[0].detach().cpu().numpy())

            # Visualize all heatmaps in a grid (3×5) and save to a file
            # vmin and vmax control the color mapping range
            imgify(heatmaps, vmin=-1, vmax=1, grid=(3, 5)).save('vit_heatmap.png')