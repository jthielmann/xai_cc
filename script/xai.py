#!/usr/bin/env python3
"""
Minimal XAI CLI with CRP for dataset | cluster | single modes.

Usage examples:
  conda run -n hest-xai python script/xai.py --from-manifest out/<run_id>/manifest.json --method crp --mode dataset
  conda run -n hest-xai python script/xai.py --from-manifest out/<run_id>/manifest.json --method crp --mode cluster
  conda run -n hest-xai python script/xai.py --from-manifest out/<run_id>/manifest.json --method crp --mode single --image /path/to/tile.png

PCX is tracked as a separate TODO in xai.md and intentionally not implemented here to keep this compact.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# CRP/zennit
from zennit.composites import EpsilonPlusFlat
from zennit.torchvision import ResNetCanonizer, VGGCanonizer
from crp.helper import get_layer_names
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept

# UMAP & GMM for cluster mode
from umap import UMAP
from sklearn.mixture import GaussianMixture

# Local utilities
from script.model.lit_model import load_model2
from script.data_processing.data_loader import get_dataset


def _load_manifest_and_cfg(path: Path) -> Tuple[dict, dict, Path]:
    m = json.loads(Path(path).read_text())
    run_dir = path.parent
    cfg_path = Path(m.get("config_resolved_path") or run_dir / "config.json")
    cfg = json.loads(Path(cfg_path).read_text())
    # ensure out_path is the run dir for downstream save paths
    cfg["out_path"] = str(run_dir)
    return m, cfg, run_dir


def _genes_from_cfg(cfg: dict) -> List[str]:
    g = cfg.get("genes") or []
    if not g:
        return []
    # Support nested chunks; flatten
    if g and isinstance(g[0], list):
        return [x for chunk in g for x in chunk]
    return list(g)


def _pick_samples(cfg: dict) -> List[str]:
    s = []
    for k in ("train_samples", "val_samples", "test_samples"):
        v = cfg.get(k)
        if isinstance(v, list):
            s.extend(v)
    # dedupe, preserve order
    seen = set()
    out = []
    for x in s:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def _simple_eval_transforms(cfg: dict):
    import torchvision.transforms as T
    size = int(cfg.get("image_size", 224))
    mean = cfg.get("mean", [0.7406, 0.5331, 0.7059])
    std = cfg.get("std",  [0.1651, 0.2174, 0.1574])
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def _composite_and_layer(model: torch.nn.Module):
    enc = getattr(model, "encoder", None)
    if enc is None:
        raise RuntimeError("Model has no attribute 'encoder'.")
    # Pick canonizer based on encoder type; fallback to ResNet canonizer
    canonizers = [ResNetCanonizer()]
    if enc.__class__.__name__.lower().startswith("vgg"):
        canonizers = [VGGCanonizer()]
    composite = EpsilonPlusFlat(canonizers=canonizers)
    layer_type = enc.__class__
    layer_name = get_layer_names(model, [layer_type])[-1]
    return composite, layer_name


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_crp_dataset(cfg: dict, out_dir: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model2(cfg)
    model.to(device).eval()
    composite, layer_name = _composite_and_layer(model)
    attribution = CondAttribution(model)
    cc = ChannelConcept()

    genes = _genes_from_cfg(cfg)
    samples = _pick_samples(cfg)
    ds = get_dataset(
        cfg["data_dir"],
        genes=genes,
        transforms=_simple_eval_transforms(cfg),
        samples=samples,
        return_floats=True,
    )
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    attributions, activations, outputs = [], [], []
    for x, _ in dl:
        x = x.to(device).requires_grad_()
        cond = [{"y": [0]}]
        attr = attribution(x, cond, composite, record_layer=[layer_name])
        attributions.append(cc.attribute(attr.relevances[layer_name], abs_norm=True).cpu())
        activations.append(attr.activations[layer_name].amax((-2, -1)).cpu())
        outputs.append(attr.prediction.detach().cpu())

    if attributions:
        A = torch.cat(attributions)
        Z = torch.cat(activations)
        O = torch.cat(outputs)
        _ensure_dir(out_dir)
        torch.save(Z, out_dir / "activations.pt")
        torch.save(A, out_dir / "attributions.pt")
        torch.save(O, out_dir / "outputs.pt")
        torch.save(torch.arange(len(ds)), out_dir / "indices.pt")
        print(f"[xai] saved {len(ds)} items → {out_dir}")
    else:
        print("[xai] no data processed; nothing saved.")


def run_crp_cluster(cfg: dict, out_dir: Path) -> None:
    # Reuse dataset-level pass, then compute UMAP + prototypes
    run_crp_dataset(cfg, out_dir)
    A = torch.load(out_dir / "attributions.pt", map_location="cpu")
    Z = torch.load(out_dir / "activations.pt", map_location="cpu")

    embedding_attr = UMAP(n_neighbors=5, random_state=123, n_jobs=1)
    X_attr = embedding_attr.fit_transform(A.detach().cpu().numpy())
    embedding_act = UMAP(n_neighbors=5, random_state=123, n_jobs=1)
    X_act = embedding_act.fit_transform(Z.detach().cpu().numpy())

    _ensure_dir(out_dir / "prototypes")
    gmm_attr = GaussianMixture(n_components=8, random_state=0).fit(A.detach().cpu().numpy())
    np.save(out_dir / "prototypes" / "attr.npy", gmm_attr.means_)
    gmm_act = GaussianMixture(n_components=8, random_state=0).fit(Z.detach().cpu().numpy())
    np.save(out_dir / "prototypes" / "act.npy", gmm_act.means_)

    np.save(out_dir / "X_attr.npy", X_attr)
    np.save(out_dir / "X_act.npy", X_act)
    print(f"[xai] cluster artifacts saved → {out_dir}")


def run_crp_single(cfg: dict, image_path: Path, out_dir: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model2(cfg)
    model.to(device).eval()
    composite, layer_name = _composite_and_layer(model)
    attribution = CondAttribution(model)

    # transforms
    T = _simple_eval_transforms(cfg)
    img = Image.open(image_path).convert("RGB")
    x = T(img).unsqueeze(0).to(device).requires_grad_()
    cond = [{"y": [0]}]
    cc = ChannelConcept()
    attr = attribution(x, cond, composite, record_layer=[layer_name])
    A = cc.attribute(attr.relevances[layer_name], abs_norm=True)[0].detach().cpu()

    import zennit.image as zimage
    from crp.image import imgify
    a = A.sum(0)
    a = a / (a.abs().max() + 1e-8)
    heat = zimage.imgify(a, symmetric=True, vmin=-1, vmax=1, cmap="coldnhot")
    base = imgify(x[0].detach().cpu())
    from PIL import Image as PILImage
    overlay = PILImage.blend(base.convert("RGBA"), heat.convert("RGBA"), alpha=0.5)

    _ensure_dir(out_dir)
    overlay.save(out_dir / (image_path.stem + "_crp_overlay.png"))
    print(f"[xai] single-image CRP saved → {out_dir}")


def main():
    p = argparse.ArgumentParser(description="XAI runner (compact)")
    p.add_argument("--from-manifest", required=True, help="Path to out/<run_id>/manifest.json")
    p.add_argument("--method", choices=["crp", "pcx"], default="crp")
    p.add_argument("--mode", choices=["dataset", "cluster", "single"], default="dataset")
    p.add_argument("--image", help="Path to a single image (for --mode single)")
    p.add_argument("--out-subdir", default="xai", help="Subdirectory under run dir for outputs")
    args = p.parse_args()

    m_path = Path(args.__dict__["from-manifest"]).resolve()
    manifest, cfg, run_dir = _load_manifest_and_cfg(m_path)
    out_dir = Path(run_dir) / args.out_subdir / f"{args.method}-{args.mode}"

    if args.method == "pcx":
        raise SystemExit("PCX: see xai.md TODOs; not implemented in this compact CLI.")

    if args.mode == "dataset":
        run_crp_dataset(cfg, out_dir)
    elif args.mode == "cluster":
        run_crp_cluster(cfg, out_dir)
    else:
        if not args.image:
            raise SystemExit("--image is required for --mode single")
        run_crp_single(cfg, Path(args.image), out_dir)


if __name__ == "__main__":
    main()

