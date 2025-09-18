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
