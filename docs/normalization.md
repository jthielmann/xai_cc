# Normalization refactor — implementation checklist (encoder-only)

## Suggested order to implement
1) **Single source of truth** (new `normalization.py`, remove duplicates)  
2) **Config schema** (policy fixed to `encoder`)  
3) **Transforms builder** (single place, correct order)  
4) **Defaults & guards** (assert one `Normalize`, strict encoder mapping)  
5) **Logging & reproducibility** (W&B + `normalization.json`)  
6) **Delete legacy uses** (ripgrep cleanup, remove dataset stats paths)  
8) **Docs & README** (short policy section)

> **Defaults**  
> - ImageNet/DINO stats: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`  
> - Train: `Resize/RandomCrop → color/stain aug → Occlude → ToTensor → Normalize`  
> - Val/Test: `Resize → ToTensor → Normalize`  
> - Same normalization for all splits; only augs differ.

---

## P0 — Single source of truth
- [ ] Create `script/configs/normalization.py` with:
  - [ ] `NormStats` dataclass
  - [ ] `IMAGENET_MEAN/STD`
  - [ ] `ENCODER_NORM: Dict[str, NormStats]` (e.g., `resnet50imagenet`, `dino`, `dinov2`, `dinov3`, `unimodel`, …)
  - [ ] `resolve_norm(encoder_type) -> NormStats` (no policy arg; always encoder)
- [ ] Remove `mean/std` from `script/configs/config_factory.py`
- [ ] Delete all dataset-stat code paths (constants, loaders, JSON readers)

## P0 — Config schema (encoder-only)
- [ ] In YAML/config, **remove** any `normalize.policy` and `normalize.dataset_key`
- [ ] Add a single flag for clarity (optional): `normalize: encoder` (string constant)
- [ ] Ensure `get_dataset_cfg(...)` does **not** inject `mean/std`

---

## P1 — Transforms builder (single entrypoint)
- [ ] Implement `build_transforms(cfg)` in `script/data_processing/transforms.py`
  - [ ] Read `encoder_type = cfg["encoder_type"]`
  - [ ] `stats = resolve_norm(encoder_type)`
  - [ ] **Train order:** `Resize/RandomCrop → color/stain aug → Occlude → ToTensor → Normalize(stats)`
  - [ ] **Val/Test order:** `Resize → ToTensor → Normalize(stats)`
- [ ] Make **all** dataloaders import and use this builder
- [ ] `ripgrep` for `Normalize(` and remove any usage **outside** this builder

## P1 — Encoder mapping & fallbacks
- [ ] Populate `ENCODER_NORM` for all encoders you use
- [ ] If an encoder is missing, **warn** and fall back to ImageNet stats

---

## P2 — Parity & guards
- [ ] Assert exactly **one** `Normalize` in train/val/test transforms
- [ ] Verify identical `mean/std` across splits (augmentations may differ; normalization may not)
- [ ] Ensure `Occlude` and stain/color jitter happen **before** `Normalize`

## P2 — Logging & reproducibility
- [ ] Log to W&B/config:
  - [ ] `normalize.mode: encoder`
  - [ ] `normalize.mean`, `normalize.std`, `encoder_type`
- [ ] Write `<out_path>/normalization.json` with the chosen stats for the run

---

## P3 — Delete legacy & simplify
- [ ] Remove dataset-stat utilities (`tools/compute_dataset_stats.py`, JSON files, loaders)
- [ ] Purge comments like “already normalized” and any `[0,0,0]/[1,1,1]` placeholders
- [ ] Remove `mean/std` from `dataset_config.py` and any unused schema entries

## P3 — Repo hygiene & docs
- [ ] Update README/docs:
  - [ ] Short “Normalization (Encoder-only)” section stating fixed ImageNet/DINO stats
  - [ ] Note transform order; avoid per-image z-scoring
- [ ] `ripgrep` to confirm only a single `Normalize` path remains and no dataset-stat mentions are left
