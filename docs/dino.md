# DINOv3 — Quick Commands

Training
- `conda run -n dino python script/dino_main.py --config sweeps/configs/dino`

Sanity Tests (no downloads)
- `python script/tests/test_dino_input_pipeline.py`  # verifies view routing and normalization toggle
- `python script/tests/smoke_train_dino.py`          # 1-epoch CPU fit with a dummy backbone

Visualization
- Stain normalization preview (CSV):
  - `python script/evaluation/visualize_stain_norm.py --csv /path/to/files.csv --tile-col tile --n 12 --out-dir stain_vis`
- Stain normalization preview (directory):
  - `python script/evaluation/visualize_stain_norm.py --dir /path/to/images --n 12 --out-dir stain_vis`

Notes
- To enable stain normalization during training, set `use_stain_norm: true` in your config (see below).
- To avoid double-normalization, toggle `use_hf_normalize` accordingly.

# DINOv3 Migration Checklist (Reviewed)

## Deps & Config
- [x] Pin `transformers` + `torch` versions
- [x] Enable bf16 only if GPU supports it (A100/4090+)
- [x] Add flag to skip HF processor normalization if transforms already handle it

## Backbone (ResNet → DINOv3 ViT)
- [x] Use `Dinov3Model` + `hidden_size`
- [x] Enable gradient checkpointing to save VRAM
- [x] If unfreezing blocks: set param groups (bias/norm no-decay) + lower LR for backbone

## Input Pipeline
- [x] Teacher: 2× global crops only
- [x] Student: global + local crops
- [x] Apply augs before normalization (avoid double-normalize)
- [x] Keep histology color jitter modest; consider stain norm

### Stain Normalization (Optional)
- Enable via `use_stain_norm: true`; method via `stain_norm_method: reinhard`.
- Targets in Lab space: `stain_target_means_lab` and `stain_target_stds_lab`.
- Order: runs before DINO multi-crop augmentations and before normalization.
- When `use_hf_normalize: true`, transforms skip Normalize to avoid double-normalization.

Visualizer
- Script: `script/evaluation/visualize_stain_norm.py`
- CSV example:
  - `python script/evaluation/visualize_stain_norm.py --csv /path/to/files.csv --tile-col tile --n 12 --out-dir stain_vis`
- Directory example:
  - `python script/evaluation/visualize_stain_norm.py --dir /path/to/images --n 12 --out-dir stain_vis`

## Projection Heads
- [x] Match `input_dim == hidden_size` (768/1024)
- [x] Keep last-layer gradient cancel logic
- [x] Verify temps match head output dim

## Forward Pass
- [x] Use CLS token for DINO loss
- [x] Keep patch tokens for Gram anchoring
- [x] Set `output_hidden_states=False` to save memory

## Gram Anchoring
- [x] Normalize patch tokens, detach teacher
- [x] Beware O(T²) cost: subsample tokens or use cosine-sim
- [x] Scale by token count for stable loss magnitude

Config
- `gram_anchor_weight`: scales the Gram anchoring term (default 0.0 = off).
- `gram_anchor_token_subsample`: optional cap on tokens per view to limit O(T²) cost.
  - Randomly subsamples tokens when `T > cap`.
- Uses cosine similarity via token normalization; MSE between (student, teacher) Gram matrices.

## Hyperparameters
- [x] EMA momentum per step (cosine 0.996 → 0.9995)
- [x] Teacher temp warmup; adjust with batch size
- [x] Optimizer: AdamW + cosine LR + warmup
- [x] Scale LR with effective batch size

Config
- `ema_momentum_start`, `ema_momentum_end`: cosine schedule per step.
- `teacher_temp_warmup`, `teacher_temp_final`, `warmup_teacher_temp_epochs`: passed to DINOLoss; warmup based on epoch.
- `warmup_epochs`: LR warmup steps = epochs × steps/epoch; then cosine LR per step.
- `scale_lr_with_batch`: scales `head/backbone` LR by `effective_batch_size / base_batch_size`.
- `base_batch_size`, `effective_batch_size`: optional overrides; otherwise auto-estimated.

## Mixed Precision
- [x] Use `autocast(bfloat16)`
- [x] Apply grad clipping before optimizer step
- [x] Optionally try `torch.compile` (profile first)

Config
- `precision_auto: true` picks `bf16-mixed` if supported, else `16-mixed`, else 32.
- `grad_clip_val`, `grad_clip_algo`: applied via Lightning’s gradient clipping hook.
- `use_torch_compile`, `compile_target: {model,backbone,head}`, `compile_mode`: guarded best-effort.

## High-Res Stage
- [x] Ensure positional embedding interpolation works
- [x] Run only a short stage; monitor VRAM

Usage
- Enable `run_high_res_stage: true` and set `high_res_epochs`.
- Override sizes: `high_global_crop_size`, `high_local_crop_size`.
- HF ViT interpolates pos-embeddings automatically.

## Checkpointing
- [x] Save student, teacher, and centers/temps for restart
- [x] Provide backbone-only export for downstream encoder

Notes
- `latest.pth` and `best_model.pth` include student/teacher and loss state (center/temps) for restart.
- `DINO.export_backbone(path)` saves backbone-only weights.

## Integration (Gene Pipeline)
- [x] Expose `pool={"cls","mean"}`; don’t rely on `pooler_output`
- [x] Add feature-cacher (eval mode, no dropout, same norm)
- [x] Store features in fp16/bf16 to save space

APIs (in `DINO`)
- `encode_features(x, pool={cls,mean}) -> (B,D)`
- `cache_features(dataloader, out_path, pool='cls', dtype={'float16','bfloat16','float32'})`

## Diagnostics
- [x] Log entropy, center norm, temps, EMA m, Gram loss
- [x] Validate with linear probe + NN retrieval
- [x] Watch for collapse (variance near zero)

Logged (on_step)
- `ema_m`, `teacher_entropy`, `center_norm`, `gram_loss`, `student_var`.

Tools
- Linear probe + NN retrieval: `python script/evaluation/linear_probe.py --data-dir /path/NCT-CRC-HE-100K --classes TUMOR STROMA ... --ckpt /path/to/best_model.pth`.

## Distributed / Loader
- [x] Ensure DINOLoss does cross-GPU centering/gather
- [x] Base schedules on global step, not epoch (with grad-accum)

Notes
- Optional `gather_distributed: true` uses Lightning’s `all_gather` on logits before DINOLoss.
- EMA momentum uses per-step cosine schedule; respects grad accumulation.

---

## Reference Delta — Recommended TODOs

These are implementation details from DINOv3-style references that differ from this repo. They are optional but beneficial.

- [x] Teacher eval mode: force `teacher_backbone` and `teacher_head` into `eval()` at all times to disable any dropout/BN updates explicitly.
- [x] No-decay params: exclude embedding params from weight decay (e.g., `pos_embed`, `cls_token`, `absolute_pos_embed`, `relative_position_bias_table`).
- [x] Layer-wise LR decay (LLRD): apply per-block LR decay for the backbone (config `layer_lr_decay`, e.g., 0.75).
- [x] Weight-decay schedule: cosine schedule per step (warmup → cosine) instead of constant weight decay.
- [x] Per-step teacher temperature: schedule by global step (config `teacher_temp_per_step: true`).
- [x] Aug policy parity: `use_ref_aug_policy: true` enforces DINOv3-like per-view augs.
- [x] Log schedules: logs `lr`, `weight_decay`, and `teacher_temp` per step.
- [x] Fused AdamW: `use_fused_adamw: true` (CUDA-only; falls back if unsupported).
- [x] SyncBatchNorm (optional): enable via Trainer (`use_sync_batchnorm: true`), effective in DDP.
- [x] Feature norm check: logs `cls_norm`, `patch_norm` per step (first student view) to monitor collapse.
