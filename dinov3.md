# DINOv3 gene-regression — experiment checklist

## Suggested order to run
1) ConvNeXt-T (frozen) → ViT-S (frozen) → **Best-bet A (ViT-B frozen)**  
2) **Best-bet B (ViT-B unfrozen @336)** → P1 size sweeps → ViT-L (frozen/unfrozen)  
3) Introduce mild color/blur only if stable

> Defaults (unless stated): **ImageNet norm**, histology-safe train augs (RRC, H/V flips, 90° rotations, tiny affine), **eval=Resize+Normalize**, **bf16/fp16**, size **256**.

## P0 — Fast baselines (frozen encoder, size=256)
- [ ] **ConvNeXt-T (frozen)** → regression head • **batch** start: 128–256  
  - [ ] Train: histology-safe augs  
  - [ ] Eval: resize → normalize
- [ ] **ViT-S/16 (frozen)** → regression head • **batch** start: 64–96  
  - [ ] Same augs/eval
- [ ] **ViT-B/16 (frozen)** → regression head • **batch** start: 32–48  
  - [ ] Same augs/eval
- [ ] Pick best P0 by **val Pearson mean** and promote to P1.

## Best-bet starting configs (recommended)
- [ ] **A. Fast baseline (frozen) — ViT-B/16 @256**
  - [ ] **Model**: ViT-B/16 (web), **frozen encoder**
  - [ ] **Transforms**: histology-safe train augs; eval=resize→normalize
  - [ ] **Batch**: 32–48 (bf16/fp16)
  - [ ] **Head**: 2–3 layer MLP, dropout 0.2
  - [ ] **Loss**: MSE vs Pearson mix (log both)
  - [ ] **Schedule**: 30–50 epochs, early-stop on **val Pearson mean**
- [ ] **B. Finetune candidate (unfrozen) — ViT-B/16 @336**
  - [ ] **Model**: ViT-B/16, **unfrozen**, **gradient checkpointing ON**
  - [ ] **Transforms**: same geometric augs; **color jitter off (or very mild)**
  - [ ] **Batch**: 8–16 (use **grad accumulation** to match A’s effective batch)
  - [ ] **Optimizer**: AdamW, **WD=0.05**, **grad clip=1.0**
  - [ ] **LR schedule**: **LLRD** (head ~1e-3; top block ~3e-5; decay ×0.75 per lower block), **cosine**, **warmup 5–10%**
  - [ ] **Train**: 60–100 epochs; patience 10–15
- [ ] **C. Capacity probe (frozen) — ViT-L/16 @256**
  - [ ] **Model**: ViT-L/16 (web), **frozen encoder**
  - [ ] **Batch**: 8–12
  - [ ] If it clearly beats B, consider **unfreezing last block only**

## P1 — Unfreeze & size sweeps
- [ ] **ViT-B/16 (unfrozen)**, size=256, **grad checkpointing ON** • **batch** start: 16–24  
  - [ ] Keep color jitter off or very mild  
  - [ ] Use grad accumulation to match P0 effective batch
- [ ] **ViT-B/16 (unfrozen)**, size=336 → **batch** ≈ half of size-256 run  
- [ ] **ViT-B/16 (unfrozen)**, size=384 → **batch** ≈ one-third of size-256 run  
- [ ] **ViT-L/16 (frozen)**, size=256 • **batch** start: 8–12  
- [ ] **ViT-L/16 (unfrozen)**, size=256, **checkpointing ON** • **batch** start: 4–8  
  - [ ] Consider **8-bit Adam** to save memory

## P2 — Regularization & robustness (start from best P1)
- [ ] + **Very mild ColorJitter** (b=0.05, c=0.05, s=0.03, h=0.01; p=0.2)  
  - [ ] If val drops, remove color first
- [ ] + **GaussianBlur** (k=3, σ=0.1–1.0; p=0.2)  
- [ ] **Stability check (TTA)**: eval once with tiny color jitter; don’t report TTA as main score  
- [ ] **Size ablation (frozen encoder)**: increase to 336/384 to test “context vs capacity”

## VRAM levers (tick when applied)
- [ ] Mixed precision (**bf16/fp16**)  
- [ ] **Gradient checkpointing** for ViTs  
- [ ] **Grad accumulation** to maintain effective batch  
- [ ] **8-bit optimizers** (e.g., bitsandbytes)  
- [ ] Reduce **image size** before reducing **batch** if batch-norm stats matter (ConvNeXt)  
- [ ] **Freeze schedule**: start frozen → unfreeze top blocks → full unfreeze

## Sanity & logging
- [ ] Confirm **normalization**: ImageNet mean/std; if upstream z-scored, apply **corrective** normalize for pretrained  
- [ ] Post-transform channel **mean/std** spot-check on a batch  
- [ ] Log with this CSV header:
