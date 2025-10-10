import os
import torch
from typing import Any, Dict, Optional

from script.evaluation.crp_plotting import plot_crp
from script.evaluation.lrp_plotting import plot_lrp
from script.evaluation.pcx_plotting import plot_pcx
from script.evaluation.tri_plotting import plot_triptych_from_merge
from script.model.model_factory import get_encoder
from script.model.lit_model import load_model


class XaiPipeline:
    def __init__(self, config, run):
        self.config = config
        self.wandb_run = run
        self.model, self.model_device, self._uses_full_model = self._load_model()

    def _resolve_device(self) -> torch.device:
        dev_cfg = self.config.get("device")
        if isinstance(dev_cfg, str):
            return torch.device(dev_cfg)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        bases = []
        cfg_src = self.config.get("_config_path")
        if cfg_src:
            bases.append(os.path.dirname(cfg_src))
        model_src = self.config.get("_model_config_path")
        if model_src:
            bases.append(os.path.dirname(model_src))
        bases.append(os.getcwd())
        for base in bases:
            candidate = os.path.normpath(os.path.join(base, path))
            if os.path.exists(candidate):
                return candidate
        return os.path.normpath(os.path.join(bases[0], path)) if bases else path

    def _load_state_dict_from_path(self, path: str) -> Dict[str, Any]:
        resolved = self._resolve_path(path)
        state = torch.load(resolved, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError(f"State dict file {resolved!r} did not contain a dictionary")
        return state

    def _normalize_state_dicts(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            raw = raw["state_dict"]
        if any(k in raw for k in ("encoder", "gene_heads", "sae")):
            return {k: raw[k] for k in ("encoder", "gene_heads", "sae") if k in raw}
        return {"encoder": raw}

    def _collect_state_dicts(self) -> Optional[Dict[str, Any]]:
        # Option 1: explicit component paths
        paths = {
            "encoder": self.config.get("encoder_state_path"),
            "gene_heads": self.config.get("gene_head_state_path"),
            "sae": self.config.get("sae_state_path"),
        }
        loaded = {k: self._load_state_dict_from_path(p) for k, p in paths.items() if p}
        if loaded:
            return loaded

        # Option 2: combined file (Lightning checkpoint or bundled dict)
        bundled_path = self.config.get("model_state_path") or self.config.get("checkpoint_path")
        if not bundled_path:
            return None
        bundled = self._load_state_dict_from_path(bundled_path)
        return self._normalize_state_dicts(bundled)

    def _load_model(self):
        device = self._resolve_device()
        state_dicts = self._collect_state_dicts()

        if state_dicts:
            model = load_model(self.config, state_dicts)
            model.to(device)
            model.eval()
            return model, device, True

        # Fallback: encoder-only (sufficient for LRP/CRP minimal demos)
        encoder = get_encoder(self.config["encoder_type"])
        encoder.to(device)
        encoder.eval()
        return encoder, device, False


    def run(self):
        if self.config.get("lrp", False):
            plot_lrp(self.model, device=self.model_device, run=self.wandb_run)
        if self.config.get("crp", False):
            plot_crp(self.model, device=self.model_device, run=self.wandb_run)
        if self.config.get("pcx", False):
            if not self._uses_full_model:
                raise RuntimeError("PCX plotting requires a trained model. Provide 'model_state_path' or component state dict paths in the config.")
            plot_pcx(self.model, self.config, run=self.wandb_run)
        if self.config.get("diff", False):
            plot_triptych_from_merge(
                self.config["data_dir"],
                self.config["patient"],
                self.config["gene"],
                self.config["out_path"],
                is_online=bool(self.wandb_run),
                wandb_run=self.wandb_run,
            )
