#!/usr/bin/env python3
"""
Minimal embedding precomputation using your existing helpers:
- get_encoder(name: str) -> torch.nn.Module
- load_dataset(root: str, transform=None) -> Dataset yielding (image_tensor, path)

Notes
-----
• If you pass an `augment_fn`, it should map a PIL.Image or torch.Tensor to a
  **normalized torch.FloatTensor** exactly as the encoder expects.
• Keep it simple: one view per image. If you later want TTA, wrap this function.
• The encoder is assumed to output a single vector per image: [B, D].

Outputs
-------
Saves a .npz at OUTPUT_PATH with:
  - embeddings: float32 array of shape [N, D]
  - paths:      array of N strings (file paths)
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

# -------------------------------------------------------------------
# Replace these imports with your project paths
from script.model.model_factory import get_encoder      # <- you have this already
from script.data_processing.data_loader import get_dataset_from_config         # <- you have this already
# -------------------------------------------------------------------

# -------- defaults (edit as needed) --------------------------------
INPUT_ROOT   = "./images"                 # folder scanned by your dataset loader
OUTPUT_PATH  = "./embeddings.npz"
ENCODER_NAME = "uni"                      # any name your get_encoder understands
BATCH_SIZE   = 128
NUM_WORKERS  = 4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
AUGMENT_FN   = None   # e.g., a torchvision transform; must output normalized tensor
SEED         = 42
# -------------------------------------------------------------------


def precompute_embeddings(
    input_root: str = INPUT_ROOT,
    output_path: str = OUTPUT_PATH,
    encoder_name: str = ENCODER_NAME,
    augment_fn = AUGMENT_FN,  # callable or None
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    device: str = DEVICE,
):
    # Repro
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(device)

    # 1) Encoder
    model = get_encoder(encoder_name)
    model.eval().to(device)

    # 2) Data
    ds = get_dataset_from_config("coad", transforms=augment_fn)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # 3) Forward pass → collect embeddings
    all_embs = []
    all_paths = []

    with torch.no_grad():
        for batch in dl:
            # Support datasets that yield (x, path) or (x, y, path)
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    x, paths = batch
                else:
                    x, _, paths = batch  # ignore labels if present
            else:
                raise ValueError("Dataset must yield (image_tensor, path) or (image_tensor, label, path).")

            x = x.to(device, non_blocking=True)
            out = model(x)                 # [B, D]
            embs = out.detach().cpu().float().numpy()
            all_embs.append(embs)
            all_paths.extend(list(paths))

    embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)

    # 4) Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(output_path, embeddings=embeddings, paths=np.array(all_paths))
    print(f"Saved {output_path} | embeddings: {embeddings.shape}, files: {len(all_paths)}")


if __name__ == "__main__":
    # Runs with the defaults above. To override in code:
    # precompute_embeddings(input_root="...", encoder_name="...", augment_fn=my_tfm)
    precompute_embeddings()
