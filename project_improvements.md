## Project Structure
- [ ] **Flatten into package:** `script/` → `hest/`; add `hest/cli.py` with subcommands: `train | xai | dino | eval | sweep`.

## Configuration & Artifacts
- [ ] **One config schema:** Pydantic (or dataclass+jsonschema) with strict validation; write `out/<run_id>/config_resolved.json`.
- [ ] **Single sweep path:** Move gene-chunking/param lifting to `hest/config/sweep.py` used by both train & xai.
- [ ] **Stable artifacts:** Emit `out/<run_id>/manifest.json` (paths, hashes, encoder dim, dataset split, metrics).
- [ ] **Env provenance:** Write lockfile hash(es) into the manifest; refuse train/xai mismatch unless `--force`.

## Data/Model Code
- [ ] **Module cleanup:** Split `data_loader.py` → `datasets.py`, `weights.py`, `sampling.py`; use `nn.ModuleDict` for per-gene heads.
- [ ] **Unified transforms:** One API (v1/v2 via flag), avoid double-normalize; expose `normalize: true|false`.

## Training Policy & Metrics
- [ ] **Centralize policy:** `hest/train/policy.py` decides LR groups, precision/autocast, grad clipping; reused by train & dino.
- [ ] **Unified logging:** Helper logs `{loss, MAE, Pearson r}` per-gene + per-patient with consistent keys across train/xai.
- [ ] **Manifest-driven plots:** Plot funcs take the manifest; no ad-hoc file discovery.

## Orchestration (Train ↔ XAI across envs)
- [ ] **Simple chain:** `Makefile`/`nox` target `train_and_xai`:
      - `conda run -n hest-train hest train --config ...`
      - `conda run -n hest-xai   hest xai --from-manifest out/<run_id>/manifest.json`
- [ ] **Parallel option:** Lightweight `xai-worker` (FastAPI) in `hest-xai`; `hest train` posts checkpoints/tiles to `http://localhost/.../explain`.

## Environment Management
- [ ] **Env pinning:** Generate `train.lock.yml`, `xai.lock.yml`, `dino.lock.yml` via `conda-lock` (or `pixi`).
- [ ] **Containers:** Docker Compose services `train` & `xai` with shared `./out` volume mirroring the two-env flow.

## CI & Testing
- [ ] **Tiny smoke set:** 100-tile synthetic + golden metrics; `hest test --smoke` runs end-to-end quickly.
- [ ] **Pre-commit & CI:** ruff, black, isort, mypy (lenient), pytest smoke; GitHub Actions runs lint+smoke on PRs.

## Notebooks & Docs
- [ ] **Notebook hygiene:** Move to `/notebooks`; each notebook shells to the CLI; figures saved only via manifest paths.
- [ ] **Docs at a glance:** Trim to one-page `ARCHITECTURE.md` + ASCII flow (train → manifest → xai) and link orchestration modes.