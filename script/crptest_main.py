from pathlib import Path

import torch
from tqdm import tqdm
import zennit.image as zimage
from crp.attribution import CondAttribution
from zennit.composites import EpsilonPlusFlat
from zennit.torchvision import ResNetCanonizer

from script.data_processing.data_loader import get_dataset_from_config
from script.data_processing.image_transforms import get_transforms
from script.evaluation.eval_helpers import collect_state_dicts
from script.main_utils import parse_yaml_config
from script.model.lit_model import load_lit_regressor

MODEL_DIR = Path(
    "../models/enc-dino_genes_id-08bba863d4_genes_id-08bba863d4_run_name-enc_dino"
).resolve()
OUT_DIR = Path("../crp_test")
DATASET_NAME = "coad"
DATA_SPLIT = "train"
GENE_DATA_FILENAME = "gene_log1p.csv"


def select_device() -> torch.device:
    has_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    has_cuda = torch.cuda.is_available()
    if has_mps:
        return torch.device("mps")
    if has_cuda:
        return torch.device("cuda")
    raise RuntimeError(f"Device unavailable: mps={has_mps} cuda={has_cuda}")


def get_block_layers(model: torch.nn.Module, encoder: torch.nn.Module) -> list[str]:
    names = []
    for name, _ in encoder.named_modules():
        if not name:
            continue
        if not name.startswith("layer"):
            continue
        if name.count(".") != 1:
            continue
        names.append(name)
    if not names:
        raise RuntimeError("No encoder block layers found for CRP.")
    model_names = {n for n, _ in model.named_modules()}
    resolved = []
    for name in names:
        prefixed = f"encoder.{name}"
        if prefixed in model_names:
            resolved.append(prefixed)
            continue
        if name in model_names:
            resolved.append(name)
            continue
        raise KeyError(f"Layer '{name}' missing in model named_modules.")
    return resolved


def main():
    device = select_device()
    cfg = parse_yaml_config(MODEL_DIR / "config")
    ds = get_dataset_from_config(
        DATASET_NAME,
        split=DATA_SPLIT,
        gene_data_filename=GENE_DATA_FILENAME,
        transforms=get_transforms(None, split=DATA_SPLIT),
        samples = ["MISC33"],
        meta_data_dir="metadata"
    )
    if len(ds) == 0:
        raise RuntimeError(f"Dataset empty dataset_name={DATASET_NAME} split={DATA_SPLIT}")
    state_dicts = collect_state_dicts({"model_state_path": str(MODEL_DIR)})
    model = load_lit_regressor(cfg, state_dicts).eval().to(device)
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise RuntimeError("Model lacks encoder attribute.")
    layers = get_block_layers(model, encoder)
    attribution = CondAttribution(model)
    composite = EpsilonPlusFlat(canonizers=[ResNetCanonizer()])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        (OUT_DIR / layer).mkdir(parents=True, exist_ok=True)
    for idx in tqdm(range(len(ds)), desc="crp"):
        sample = ds[idx]
        x = sample[0] if isinstance(sample, (tuple, list)) else sample
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(device)
        x.requires_grad_(True)
        x = x + x.new_zeros(())
        attr = attribution(x, [{"y": [0]}], composite, record_layer=layers)
        for layer in layers:
            rel = attr.relevances.get(layer)
            if rel is None:
                keys = list(attr.relevances.keys())[:8]
                raise KeyError(f"Layer {layer} missing in relevances; keys={keys}")
            rel = rel.sum(1).detach().cpu()
            rel = torch.nan_to_num(rel, nan=0.0, posinf=1.0, neginf=-1.0)
            denom = torch.nan_to_num(rel.abs(), nan=0.0).amax(dim=(1, 2), keepdim=True)
            denom = denom.clamp_min(1e-12)
            rel = rel / denom
            img = zimage.imgify(rel, symmetric=True, cmap="coldnhot", vmin=-1, vmax=1)
            fn = OUT_DIR / layer / f"crp_{idx:04d}.png"
            img.save(fn)


if __name__ == "__main__":
    main()
