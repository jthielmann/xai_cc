from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from PIL import Image


# --- sRGB <-> Lab utilities (D65) ---

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(c <= 0.0031308, 12.92 * c, (1 + a) * (c ** (1 / 2.4)) - a)


def _rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    # expects float np array in range [0, 1]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    r_lin, g_lin, b_lin = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)
    x = 0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin
    y = 0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin
    z = 0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin
    return np.stack([x, y, z], axis=-1)


def _xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r_lin = 3.2404542 * x + -1.5371385 * y + -0.4985314 * z
    g_lin = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b_lin = 0.0556434 * x + -0.2040259 * y + 1.0572252 * z
    r = _linear_to_srgb(r_lin)
    g = _linear_to_srgb(g_lin)
    b = _linear_to_srgb(b_lin)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def _f_lab(t: np.ndarray) -> np.ndarray:
    delta = 6 / 29
    return np.where(t > delta ** 3, np.cbrt(t), (t / (3 * delta ** 2)) + (4 / 29))


def _f_inv_lab(ft: np.ndarray) -> np.ndarray:
    delta = 6 / 29
    return np.where(ft > delta, ft ** 3, 3 * delta ** 2 * (ft - 4 / 29))


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    # D65 white
    xyz = _rgb_to_xyz(rgb)
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = xyz[..., 0] / Xn, xyz[..., 1] / Yn, xyz[..., 2] / Zn
    fx, fy, fz = _f_lab(x), _f_lab(y), _f_lab(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = _f_inv_lab(fx) * Xn
    y = _f_inv_lab(fy) * Yn
    z = _f_inv_lab(fz) * Zn
    xyz = np.stack([x, y, z], axis=-1)
    return _xyz_to_rgb(xyz)


@dataclass
class ReinhardParams:
    target_means_lab: Tuple[float, float, float] = (50.0, 0.0, 0.0)
    target_stds_lab: Tuple[float, float, float] = (15.0, 5.0, 5.0)


class StainNormalizeReinhard:
    """Simple Reinhard stain normalization.

    Converts to Lab, matches image channel mean/std to target mean/std, then converts back to sRGB.
    - Input: PIL.Image (RGB) or ndarray float in [0,1]
    - Output: PIL.Image (RGB)
    """

    def __init__(self, params: ReinhardParams | None = None):
        self.params = params or ReinhardParams()

    def __call__(self, img):
        if isinstance(img, Image.Image):
            arr = np.asarray(img).astype(np.float32) / 255.0
        else:
            arr = np.asarray(img).astype(np.float32)
            if arr.max() > 1.001:
                arr = arr / 255.0

        # RGB -> Lab
        lab = rgb_to_lab(arr)
        # Compute per-channel mean/std
        im_mean = lab.reshape(-1, 3).mean(axis=0)
        im_std = lab.reshape(-1, 3).std(axis=0)
        im_std = np.where(im_std < 1e-6, 1e-6, im_std)

        tgt_mean = np.array(self.params.target_means_lab, dtype=np.float32)
        tgt_std = np.array(self.params.target_stds_lab, dtype=np.float32)

        lab_norm = (lab - im_mean) / im_std * tgt_std + tgt_mean
        # clamp Lab ranges conservatively
        lab_norm[..., 0] = np.clip(lab_norm[..., 0], 0.0, 100.0)
        lab_norm[..., 1] = np.clip(lab_norm[..., 1], -127.0, 127.0)
        lab_norm[..., 2] = np.clip(lab_norm[..., 2], -127.0, 127.0)

        # Lab -> RGB
        rgb = lab_to_rgb(lab_norm)
        rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        return Image.fromarray(rgb, mode='RGB')


class StainThenDINO:
    """Wraps a stain transform to run before a DINO multi-crop transform.

    The wrapped DINO transform returns a list of crops. We avoid Compose here and
    call DINO's transform after stain normalization.
    """

    def __init__(self, stain_tfm, dino_tfm):
        self.stain_tfm = stain_tfm
        self.dino_tfm = dino_tfm

    def __call__(self, img):
        if self.stain_tfm is not None:
            img = self.stain_tfm(img)
        return self.dino_tfm(img)

