# ST-XAI pipeline checklist

## Core training & orchestration
- [x] Lightning trainer wiring (W&B logger, callbacks, profiler stubs)
- [x] Run naming → output dir creation
- [x] Train/val/test flow + summary dump
- [x] EarlyStopping monitor key matches logged metric
- [x] Deterministic runs (seed_everything, deterministic=True)

## Devices & performance
- [x] AMP on GPU, 32-bit on CPU/MPS
- [x] Robust accelerator/device selection (CUDA vs MPS vs CPU)

## Data module & splits
- [ ] Enforce patient-stratified splits + log split manifests
- [ ] Tech-aware settings (Visium vs Xenium) for patching/alignment

## LR tuning
- [x] Per-component lr_find loop
- [x] Restore state between tuning steps
- [x] Fix variable name in error (`g` → `key`)
- [x] Skip encoder tuning when `freeze_encoder=True` (verify)

## Model integration
- [x] `get_model(config)` + `model.update_lr(lrs)`
- [x] Expose `gene→output_idx` mapping

## Logging & W&B
- [x] Fix abbrev typo: loss_fn_switch → loss_fn_switch
- [x] Log dataset meta, splits, gene panel
- [x] Save config + best checkpoint as W&B artifacts

## Spatial predictions & plots
- [x] Write preds back to spatial df
- [x] Triptych plots + W&B upload
- [x] Confirm coordinate frame; choose global vs per-patient vmin/vmax
- [x] Save per-patient parquet (x,y,label,pred,diff)

## Explainability
- [ ] Config-gated XAI runner (IG/Grad-CAM/Zennit/CRP)
- [ ] Use gene→output_idx for targets
- [ ] Save overlays + attribution stats; optional concept clustering

## HEST integration
- [ ] Check data size of SKCM, IDC, LUAD
- [ ] Adapter to HEST-Library (query → AnnData + patches)
- [ ] Standardize normalization & gene panel selection
- [ ] Verify/record alignment; log provenance
- [ ] Implement dataset filter by cancer type (SKCM, IDC, LUAD)
- [ ] Run baseline model on SKCM subset → log metrics + histograms
- [ ] Repeat for IDC subset → compare to SKCM
- [ ] Repeat for LUAD subset → compare to SKCM + IDC
- [ ] Check overlap in gene panel across subsets; flag missing genes
- [ ] Save per-subset manifests (patients, samples, gene coverage)
- [ ] W&B sweep: compare training curves across SKCM/IDC/LUAD
- [ ] Export per-subset results CSV (val loss, mean Pearson, per-gene Pearson)

## Repro & hygiene
- [x] Memory cleanup (gc/empty_cache)
- [x] Global seeding + determinism settings
- [x] Log package versions
- [x] Validate config/shape consistency

## UX & failure modes
- [x] Clear errors: missing gene, output<genes, no GPU
- [x] `--dry-run` to sanity-check data + single forward