# Project Improvements — Refined Plan

This version integrates prioritization, definitions of done, schema/versioning, and reproducibility details while keeping your original intent and structure. It avoids introducing packaging or command-line tooling changes; keep using the existing `script/` entrypoints.

## Priorities
- P0: Config schema + resolution; Manifest + schema + env policy; Smoke test + CI.
- P1: Orchestration (train ↔ xai); Unified transforms + dataset registry; Logging/metrics unification.
- P2: Parallel XAI worker; Containers/Compose; Notebook hygiene + docs.

## Project Structure
- [ ] Keep `script/` as the primary entrypoints (`script/train.py`, `script/xai.py`, `script/eval.py`, `script/sweep.py`, `script/manifest.py`).
- [ ] Move shared logic into `script/common/` (imported via repo-relative imports).
- [ ] Remove ad-hoc code from scripts; scripts only parse args and call functions in `script/common/*`.
- DoD:
  - Running `python script/train.py --config ...` works from repo root.
  - All scripts import shared code from `script/common/` (run from repo root).
  - No new packaging or external entrypoints are introduced.

## Configuration & Artifacts
- [ ] One Pydantic v2 `Config` with JSON Schema emitted to `schema/config.schema.json`.
- [ ] Config resolution order: defaults < file < env < script args < sweep.
- [ ] Persist to `out/<run_id>/{config_raw.json, config_resolved.json}`.
- [ ] Single sweep generator in `script/common/sweep.py`; `python script/sweep.py --grid ...` writes configs under `out/sweeps/<tag>/`.
- [ ] Stable manifest with versioning and schema at `schema/manifest.v1.json`; `python script/manifest.py validate <path>` validates.
- [ ] Env policy: `--env-policy strict|warn|ignore`; record lockfile name+hash in manifest and fail on `strict` mismatch unless `--force`.
- DoD:
  - Loading any config validates against the JSON Schema.
  - Manifest writes and validates; both config files are created.
  - Env policy respected with clear error/warn messaging.

## Data/Model Code
- [ ] Split `data_loader.py` into `datasets.py`, `weights.py`, `sampling.py`.
- [ ] Dataset registry helper (e.g., `script/common/datasets.py`) to return dataset objects; avoid ad-hoc conditionals.
- [ ] Model heads in `script/common/heads.py` with a small `HeadRegistry` keyed by gene id; use `nn.ModuleDict` for per-gene heads.
- [ ] Unified transforms API with `version: v1|v2` and `normalize: true|false`; record `mean/std` in manifest.
- DoD:
  - One place to construct datasets and transforms.
  - Heads selected by registry; no scattered per-gene branching.

## Training Policy & Metrics
- [ ] Centralize policy in `script/common/train_policy.py`: optimizer groups, AMP/precision, grad clipping, grad accumulation, seeding, determinism flags.
- [ ] Unified logging helpers emit consistent keys used by train & xai.
- [ ] Standard metric keys:
  - `loss/{train,val}`
  - `mae/{train,val}`
  - `pearson/per_gene/<gene>`
  - `mae/per_patient`
- [ ] Writers: JSONL per-step + CSV epoch summary; optional `--log-mlflow` later.
- DoD:
  - Both train and xai scripts use the same logging helper and keys.
  - Determinism/seed applied and recorded.

## Orchestration (Train ↔ XAI)
- [ ] Simple chain via `Makefile`/`nox` target `train_and_xai`:
  - `conda run -n hest-train python script/train.py --config ...`
  - `conda run -n hest-xai   python script/xai.py --from-manifest out/<run_id>/manifest.json`
- [ ] `script/xai.py` accepts `--from-manifest` only; no ad-hoc file discovery.
- [ ] Optional parallel mode: lightweight `xai-worker` (FastAPI) in `hest-xai`; bounded queue, retry/backoff, optional auth token.
- DoD:
  - Chain target runs end-to-end using manifest only.
  - Parallel mode gated by a flag and documented.

## Environment Management
- [ ] Generate lockfiles per env via `conda-lock` (or `pixi`): `train.lock.yml`, `xai.lock.yml`, `dino.lock.yml`.
- [ ] `make env-[train|xai|dino]` to create/update envs, printing lock hash.
- [ ] Docker Compose with services `train` and `xai`, shared `./out` volume to mirror two-env flow.
- DoD:
  - Lock hashes recorded in manifest; `strict` policy catches mismatches.
  - Compose can run the train→xai pipeline locally.

## CI & Testing
- [ ] Tiny smoke dataset (~100 tiles) and golden metric ranges; `pytest -m smoke` and an optional `script/test_smoke.py` wrapper run fast.
- [ ] Pre-commit: ruff, black, isort, mypy (lenient), nbstripout for notebooks.
- [ ] GitHub Actions: lint + smoke on PRs.
- DoD:
  - CI is green on a clean checkout; smoke completes under a few minutes.

## Notebooks & Docs
- [ ] Move notebooks to `/notebooks`; notebooks call `python script/*.py ...` (no embedded training logic); save figures via manifest paths.
- [ ] One-page `ARCHITECTURE.md` with ASCII flow (train → manifest → xai) and orchestration modes.
- DoD:
  - Notebooks are reproducible and do not embed hidden state.

## Reproducibility
- [ ] Single `seed` applied to Python/NumPy/Torch; set and record Torch determinism flags.
- [ ] Record Git provenance in manifest: `git.commit`, `git.is_dirty`, and remote URL.
- [ ] Record versions: `python`, `torch`, `cuda`.

## Manifest Contract (v1)
- Fields:
  - `manifest_version`, `run_id`, `created_at`
  - `git.{commit,is_dirty,remote}`
  - `python`, `torch`, `cuda`
  - `env_lock.{name,hash}`
  - `dataset.{name,version,split,hashes}`
  - `encoder.{arch,dim,weights}`
  - `checkpoints: [paths]`
  - `config_path`, `config_resolved_path`
  - `artifacts.{metrics,figures,plots}`
  - `transforms.{version,normalize,mean,std}`
- Validation:
  - JSON Schema at `schema/manifest.v1.json`.
  - Validate via `python script/manifest.py validate <path>`.

## Caching
- [ ] Preprocessing cache under `out/cache` keyed by content hash (dataset + transform + version).

## Error Handling
- [ ] Consistent exit codes; clear remediation text for config/env mismatches.

## Deprecation Policy
- [ ] If renaming scripts, keep the old names as shims with warnings for one release, then remove in the next.

## Clarifications Needed
- Config library: Confirm Pydantic v2 (not Hydra).
- Environments: Conda as default; ok to add Docker/Compose for parity?
- XAI worker: Same GPU box or separate host? Authentication requirements?
- Metrics: Which per-gene/per-patient metrics are authoritative for gating?
- Datasets: Need a versioned registry or are static paths sufficient?

## Next Steps
- Implement P0 items (config+manifest+env policy, smoke+CI, shared code under `script/common/` while keeping `script/` entrypoints).
- I can add stubs for `script/manifest.py` and `script/sweep.py` to operationalize validation and sweeps.
