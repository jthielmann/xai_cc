# Common beginner mistakes with encoder preprocessing (and how to avoid them)

## Normalization / scaling
- Using the wrong mean/std for the encoder (e.g., ImageNet stats for CLIP, or vice-versa).
- Skipping normalization entirely when loading a pretrained backbone (most expect specific mean/std).
- Normalizing **twice** (e.g., `ToTensor()` scales to [0,1] and you also divide by 255 again).
- Normalizing **before** converting to float (integer division/overflow artifacts).
- Mixing per-image standardization with dataset-level mean/std (quiet distribution shift).
- Computing dataset mean/std on a **mixed** pool (train+val+test) → leakage.
- Computing stats **after heavy augmentations** (changes what the model sees at test time).
- Using train-set mean/std for val/test in a **domain-mismatched** scenario; sometimes better to stick to the encoder’s canonical preprocessing.
- Forgetting to de-normalize for visualization and then “fixing” the pipeline to match the weird plot.
- Assuming grayscale images don’t need special handling (should you replicate channels or adapt the encoder?).

## Color / channel order / dtype
- Confusing RGB vs BGR (e.g., OpenCV default is BGR).
- Channel order mixups (HWC vs CHW).
- Sending uint8 tensors to the model (not converting to float32).
- JPEGs with embedded color profiles drifting hues after certain loaders (keep a consistent loader).
- Accidentally converting to HSV/other color spaces during augmentations and not converting back.

## Resize / crop / geometry
- Warping 224×224 directly instead of the typical 256→center-crop-224 (or the encoder’s expected recipe).
- Using different interpolation between train and eval (e.g., bilinear vs bicubic).
- Forgetting antialiasing when downscaling, creating aliasing artifacts.
- For ViTs: changing input resolution but **not** interpolating position embeddings (or using a loader that does it for you).
- Applying random crops that systematically remove informative regions (tile/patch tasks are sensitive).

## Augmentations (order & leakage)
- Normalizing **before** geometric/color augs that expect [0,1] (keep the usual: convert → scale → augs → normalize).
- Using strong color jitter + heavy blur on medical imagery without checking label stability.
- Applying **any** training augmentation to val/test splits.
- Computing normalization stats **after** augmentations (should be on raw training images only).

## BatchNorm / model modes
- Leaving frozen encoders in `train()` so BatchNorm running stats drift.
- Unfreezing encoders but using tiny batch sizes → unstable BatchNorm; consider SyncBN/GroupNorm or gradient accumulation.
- Forgetting to call `model.eval()` at validation/test time.

## Framework specifics (torchvision/timm)
- Assuming all timm models use ImageNet-1k preprocessing (CLIP/SAM/IN22k variants have different recipes).
- Mixing PIL and tensor transforms incorrectly (torchvision v2 functional transforms expect consistent types).
- Relying on legacy `transforms.ToTensor()` behavior and not realizing it scales to [0,1].
- Using `Normalize` with the wrong value scale (std/mean meant for [0,1] applied to [0,255] images).

## Dataset splits & bookkeeping
- Recomputing mean/std after every data refresh and silently changing the pipeline (non-reproducible results).
- Different normalization between experiments without logging it (results become incomparable).
- Accidentally normalizing targets (e.g., gene expression) differently between train and val (or using statistics that include val/test).

## Targets (for regression like gene expression)
- Mixing log1p/standard-scaling/rank-norm across splits or experiments without recording which was used.
- Standardizing targets with statistics that include validation/test (leakage).
- Forgetting to persist the **exact** target scaler used at train time and reusing at inference.

## Data loaders & I/O quirks
- Random resizing/cropping not seeded → non-deterministic validation.
- Using different tile extraction logic for train vs eval (e.g., center crop vs random crop) without realizing.
- Accidentally shuffling labels when building custom datasets (off-by-one merges, misaligned CSV joins).

## Mixed precision & dtype
- Normalizing in float16 and accumulating rounding error (do preprocessing in float32; then cast if needed).
- Converting to float64 by accident (slow, memory-heavy) with no benefit.

## Sanity checks people skip
- Not printing a batch’s **min/max/mean/std** *after the full transform pipeline*.
- Not visualizing 16 random samples **after** transforms (including color augs) to catch obvious mistakes.
- Not asserting input shapes (N,C,H,W) and value ranges before the forward pass.
- Not unit-testing that train/val pipelines are **identical** except for augmentations and `train()/eval()` mode.

## Tiling / histology-specific pitfalls
- Computing normalization on background-heavy tiles → biased means (filter or weight by tissue content).
- Inconsistent magnification/resolution across cohorts without rescaling to a common physical scale.
- Whitening/background removal altering color distribution after you tuned normalization.
- Leaking patient/site across train/val when building tiles (patient-level splits are safer than tile-level).

---

### Quick self-audit (fast checklist)
- Do I use the **exact** preprocessing recipe recommended for my encoder?
- Are train/val/test transforms identical except for augmentations and `model.eval()`?
- Are inputs float32 in [0,1] **before** `Normalize(mean,std)` (if that’s the expected recipe)?
- Are mean/std computed **only on the training set** (and saved alongside the model)?
- Did I visually inspect post-transform images and print tensor stats?
- For ViTs: did I handle positional embeddings when changing resolution?

