import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image
import numpy as np
from script.data_processing.stain_normalization import StainNormalizeReinhard, ReinhardParams


def run():
    print("[stain] apply Reinhard stain normalization to a dummy image")
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), mode='RGB')
    norm = StainNormalizeReinhard(ReinhardParams())
    out = norm(img)
    assert isinstance(out, Image.Image)
    print("[stain] output is a PIL.Image with same size:", out.size)


if __name__ == '__main__':
    run()

