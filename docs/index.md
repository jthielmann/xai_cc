# HEST — Project Docs

Welcome! This site documents the HEST project: supervised gene-expression prediction from histology tiles, plus optional DINO self-supervised pretraining.

- **Start here:** [Architecture Overview](architecture.md)
- **Repo:** See `script/` for entrypoints and `script/common/` for shared logic.

## What’s in this repo?

- **Train pipeline:** `script/train.py` → Lightning training, metrics, artifacts in `out/<run_id>/`
- **XAI pipeline:** `script/xai.py` → runs from a manifest (`--from-manifest`)
- **Common code:** `script/common/` (datasets, transforms, heads, train policy, logging)
- **Configs & schema:** JSON Schema under `schema/`, resolved configs saved per run
- **Envs:** Conda lockfiles per env (`train.lock.yml`, `xai.lock.yml`, `dino.lock.yml`)
- **CI/Smoke:** Tiny dataset + tests to validate the pipeline boots

## Quick Start

```bash
# (optional) create envs from your lockfiles
make env-train
make env-xai

# Train (example)
conda run -n hest-train python script/train.py --config sweeps/configs/debug

# Run XAI on the produced manifest (example)
conda run -n hest-xai python script/xai.py --from-manifest out/<run_id>/manifest.json

## DINOv3 Pretraining

- Guide: see `docs/dino.md` for the migration checklist and training tips.
- Normalization: toggle `use_hf_normalize` to avoid double-normalization when using HuggingFace processors vs. transform `Normalize`.
- Stain normalization (optional): enable with `use_stain_norm: true` (method `reinhard`).
- Visualizer: preview stain normalization via `script/evaluation/visualize_stain_norm.py`.

Example run

```bash
conda run -n dino python script/dino_main.py --config sweeps/configs/dino
```

Enable stain normalization

- In a sweep config (try both or force on):
  - Try both: set `parameters.use_stain_norm.values: [false, true]` (already in `sweeps/configs/dino`).
  - Force on: set `parameters.use_stain_norm.values: [true]`.
- In a single-run config (no sweep):
  - Use `parameters.use_stain_norm.value: true` and optionally set
    `parameters.stain_target_means_lab.value: [50.0, 0.0, 0.0]` and
    `parameters.stain_target_stds_lab.value: [15.0, 5.0, 5.0]`.
