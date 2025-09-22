import argparse
import os
import random
import sys
from pathlib import Path
from typing import List

from PIL import Image
import pandas as pd

# Ensure repo root on path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.data_processing.stain_normalization import (
    StainNormalizeReinhard,
    ReinhardParams,
)


def find_images_in_dir(root: Path, exts=(".tif", ".tiff", ".png", ".jpg", ".jpeg")) -> List[Path]:
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return files


def concat_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    w = left.width + right.width
    h = max(left.height, right.height)
    out = Image.new("RGB", (w, h), color=(255, 255, 255))
    out.paste(left, (0, 0))
    out.paste(right, (left.width, 0))
    return out


def main():
    p = argparse.ArgumentParser(description="Visualize before/after stain normalization on tiles.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=str, help="CSV with a tile path column (default 'tile').")
    src.add_argument("--dir", type=str, help="Directory to search for images recursively.")
    p.add_argument("--tile-col", type=str, default="tile", help="Column name with image paths in CSV.")
    p.add_argument("--n", type=int, default=8, help="Number of samples to visualize.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    p.add_argument("--out-dir", type=str, default="stain_vis", help="Directory to save side-by-side images.")
    p.add_argument("--resize", type=int, default=256, help="Resize shorter side for display; 0 to keep original size.")
    p.add_argument("--means", type=float, nargs=3, default=[50.0, 0.0, 0.0], help="Target Lab means for Reinhard.")
    p.add_argument("--stds", type=float, nargs=3, default=[15.0, 5.0, 5.0], help="Target Lab stds for Reinhard.")
    args = p.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect tile paths
    tiles: List[Path]
    if args.csv:
        df = pd.read_csv(args.csv)
        if args.tile_col not in df.columns:
            raise ValueError(f"Column '{args.tile_col}' not found in {args.csv}")
        tiles = [Path(p) for p in df[args.tile_col].tolist()]
    else:
        tiles = find_images_in_dir(Path(args.dir))

    if len(tiles) == 0:
        raise SystemExit("No images found.")

    # Sample N
    tiles = random.sample(tiles, min(args.n, len(tiles)))

    # Build stain normalizer
    normalizer = StainNormalizeReinhard(
        ReinhardParams(target_means_lab=tuple(args.means), target_stds_lab=tuple(args.stds))
    )

    # Helper to resize for display
    def _resize(img: Image.Image) -> Image.Image:
        if args.resize and args.resize > 0:
            w, h = img.size
            if w <= h:
                new_w = args.resize
                new_h = int(h * (new_w / w))
            else:
                new_h = args.resize
                new_w = int(w * (new_h / h))
            return img.resize((new_w, new_h), Image.BILINEAR)
        return img

    # Process and save
    for i, path in enumerate(tiles):
        try:
            src_img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Skipping {path} — open failed: {e}")
            continue

        dst_img = normalizer(src_img)
        if args.resize and args.resize > 0:
            src_img = _resize(src_img)
            dst_img = _resize(dst_img)

        vis = concat_side_by_side(src_img, dst_img)
        out_path = out_dir / f"stain_vis_{i:03d}.jpg"
        try:
            vis.save(out_path, quality=95)
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Failed to save {out_path}: {e}")


if __name__ == "__main__":
    main()

