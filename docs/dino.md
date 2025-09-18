# DINOv3 Migration Checklist (Reviewed)

## Deps & Config
- [ ] Pin `transformers` + `torch` versions
- [ ] Enable bf16 only if GPU supports it (A100/4090+)
- [ ] Add flag to skip HF processor normalization if transforms already handle it

## Backbone (ResNet → DINOv3 ViT)
- [ ] Use `Dinov3Model` + `hidden_size`
- [ ] Enable gradient checkpointing to save VRAM
- [ ] If unfreezing blocks: set param groups (bias/norm no-decay) + lower LR for backbone

## Input Pipeline
- [ ] Teacher: 2× global crops only
- [ ] Student: global + local crops
- [ ] Apply augs before normalization (avoid double-normalize)
- [ ] Keep histology color jitter modest; consider stain norm

## Projection Heads
- [ ] Match `input_dim == hidden_size` (768/1024)
- [ ] Keep last-layer gradient cancel logic
- [ ] Verify temps match head output dim

## Forward Pass
- [ ] Use CLS token for DINO loss
- [ ] Keep patch tokens for Gram anchoring
- [ ] Set `output_hidden_states=False` to save memory

## Gram Anchoring
- [ ] Normalize patch tokens, detach teacher
- [ ] Beware O(T²) cost: subsample tokens or use cosine-sim
- [ ] Scale by token count for stable loss magnitude

## Hyperparameters
- [ ] EMA momentum per step (cosine 0.996 → 0.9995)
- [ ] Teacher temp warmup; adjust with batch size
- [ ] Optimizer: AdamW + cosine LR + warmup
- [ ] Scale LR with effective batch size

## Mixed Precision
- [ ] Use `autocast(bfloat16)`
- [ ] Apply grad clipping before optimizer step
- [ ] Optionally try `torch.compile` (profile first)

## High-Res Stage
- [ ] Ensure positional embedding interpolation works
- [ ] Run only a short stage; monitor VRAM

## Checkpointing
- [ ] Save student, teacher, and centers/temps for restart
- [ ] Provide backbone-only export for downstream encoder

## Integration (Gene Pipeline)
- [ ] Expose `pool={"cls","mean"}`; don’t rely on `pooler_output`
- [ ] Add feature-cacher (eval mode, no dropout, same norm)
- [ ] Store features in fp16/bf16 to save space

## Diagnostics
- [ ] Log entropy, center norm, temps, EMA m, Gram loss
- [ ] Validate with linear probe + NN retrieval
- [ ] Watch for collapse (variance near zero)

## Distributed / Loader
- [ ] Ensure DINOLoss does cross-GPU centering/gather
- [ ] Base schedules on global step, not epoch (with grad-accum)