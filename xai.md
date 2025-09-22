# XAI Integration — Minimal Checklist

Goal: Add an explainability (XAI) stage after training using a separate Conda environment.

## 1) Separate Conda Environment
- [ ] Copy `conda_environments/crp.yml` → `conda_environments/xai.yml`.
- [ ] Edit `xai.yml`: set `name: hest-xai` and remove any `prefix:` line.
- [ ] Create env: `conda env create -f conda_environments/xai.yml`
- [ ] Update env (later): `conda env update -n hest-xai -f conda_environments/xai.yml`
- Notes: Current CRP stack pins Python 3.9 / torch 1.13 / numpy<=1.23 in `crp.yml`. Keep XAI separate from training (`main.py`/`lit_train.py`).

## 2) Contract: Run Input
- [ ] Train normally: `conda run -n hest-train python script/main.py --config <cfg>`
- [ ] Ensure outputs under `out/<run_id>/` include checkpoints (and a manifest if available).
- [ ] Validate manifest (if present): `conda run -n hest-train python script/manifest.py validate out/<run_id>/manifest.json`

## 3) XAI Entry Point (CLI)
- [ ] Add `script/xai.py` that:
  - [ ] Accepts `--from-manifest out/<run_id>/manifest.json`.
  - [ ] Loads model checkpoint + resolved config; reconstructs dataset (same transforms).
  - [ ] Selects target layer (e.g., `get_composite_layertype_layername_lightning(model, target_layer_name="encoder")`).
  - [ ] Runs CRP/attribution with `CondAttribution` and saves to `out/<run_id>/xai/`:
        `activations.pt`, `attributions.pt`, `outputs.pt`, `indices.pt`, `prototypes/`, and a summary figure (e.g., `result.png`).
  - [ ] Optionally add a quick gradient map path via `script/evaluation/relevance.py` for sanity checks.
- Temporary alternative (until `script/xai.py` exists):
  - [ ] Run CRP prototype pipeline: `conda run -n hest-xai python script/evaluation/cluster_explanations.py`
        (adjust constants inside for `data_dir` and `out_path` to point to the run.)

## 3b) Methods & Modes (TODOs)
- [ ] CRP method:
  - [ ] Dataset mode: explain all tiles; cache `{activations,attributions,outputs,indices}`.
  - [ ] Cluster mode: UMAP + GMM prototypes; save `prototypes/` and summary figures.
  - [ ] Single-image mode: run on one tile path or (patient,tile) id; emit heatmap overlay.
- [ ] PCX method:
  - [ ] Dataset mode: prototype/patch-based concept explanations across the run.
  - [ ] Cluster mode: concept discovery over embeddings; per-cluster exemplars.
  - [ ] Single-image mode: per-image PCX visualization/overlay.
- [ ] CLI surface (either subcommands or flags):
  - Subcommands: `xai.py crp|pcx --from-manifest ... --mode dataset|cluster|single [--image <path>]`
  - Or flags: `xai.py --method crp|pcx --mode dataset|cluster|single [--image <path>]`

## 4) Makefile Wiring (optional, simple)
- [ ] `env-xai`: `conda env create -f conda_environments/xai.yml`
- [ ] `xai`: `conda run -n hest-xai python script/xai.py --from-manifest out/<run_id>/manifest.json`
- [ ] `train_and_xai`: chain train → xai using the two envs.

## 5) Minimal Run Example
- [ ] Create envs: `make env-train` and `make env-xai` (or use the conda commands above).
- [ ] Train: `conda run -n hest-train python script/main.py --config sweeps/configs/debug`
- [ ] XAI:  `conda run -n hest-xai python script/xai.py --from-manifest out/<run_id>/manifest.json`

## 6) Notes & Tips
- [ ] Keep zennit/crp isolated to `hest-xai` to avoid version conflicts with Lightning/train.
- [ ] XAI outputs can be large; ensure free disk space.
- [ ] If you use lockfiles, generate and track `xai.lock.yml` separately from train/dino.
