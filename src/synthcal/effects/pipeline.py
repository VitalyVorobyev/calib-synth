"""Image effects pipeline (v0).

The pipeline is deterministic given an explicit RNG:
    blur -> noise -> clip -> quantize
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from synthcal.effects.config import EffectsConfig

try:
    from scipy.ndimage import gaussian_filter as _gaussian_filter  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    _gaussian_filter = None


def _gaussian_kernel1d(sigma: float, *, truncate: float = 3.0) -> np.ndarray:
    radius = int(math.ceil(float(truncate) * float(sigma)))
    if radius <= 0:
        return np.array([1.0], dtype=np.float32)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    k = k / float(np.sum(k))
    return k.astype(np.float32, copy=False)


def _convolve1d_nearest(img: np.ndarray, kernel: np.ndarray, *, axis: int) -> np.ndarray:
    if img.ndim != 2:
        raise ValueError("img must be 2D")
    if kernel.ndim != 1:
        raise ValueError("kernel must be 1D")
    radius = int((kernel.shape[0] - 1) // 2)
    if radius <= 0:
        return img.astype(np.float32, copy=False)

    img_f = img.astype(np.float32, copy=False)
    if axis == 0:
        padded = np.pad(img_f, ((radius, radius), (0, 0)), mode="edge")
        out = np.empty_like(img_f, dtype=np.float32)
        for r in range(img_f.shape[0]):
            window = padded[r : r + 2 * radius + 1, :]
            out[r, :] = kernel @ window
        return out
    if axis == 1:
        padded = np.pad(img_f, ((0, 0), (radius, radius)), mode="edge")
        out = np.empty_like(img_f, dtype=np.float32)
        for c in range(img_f.shape[1]):
            window = padded[:, c : c + 2 * radius + 1]
            out[:, c] = window @ kernel
        return out
    raise ValueError("axis must be 0 or 1")


def _gaussian_blur(img: np.ndarray, *, sigma: float) -> np.ndarray:
    if _gaussian_filter is not None:
        return _gaussian_filter(img, sigma=float(sigma), mode="nearest").astype(
            np.float32, copy=False
        )
    k = _gaussian_kernel1d(float(sigma))
    out = _convolve1d_nearest(img, k, axis=1)
    out = _convolve1d_nearest(out, k, axis=0)
    return out.astype(np.float32, copy=False)


def apply_effects(
    img_u8: Any,
    cfg: EffectsConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply blur + noise + quantization to a grayscale image.

    Parameters
    ----------
    img_u8:
        Input image, uint8 array of shape `(H, W)`.
    cfg:
        Effects configuration.
    rng:
        RNG used for noise. Required when `cfg.noise_sigma > 0` to keep results deterministic.

    Returns
    -------
    np.ndarray
        Output image, uint8 array of shape `(H, W)`.
    """

    img = np.asarray(img_u8)
    if img.dtype != np.uint8:
        raise ValueError(f"img_u8 must have dtype uint8, got {img.dtype}")
    if img.ndim != 2:
        raise ValueError(f"img_u8 must have shape (H, W), got {img.shape}")

    if not cfg.enabled:
        return img.copy()

    blur_sigma_px = float(cfg.blur_sigma_px)
    noise_sigma = float(cfg.noise_sigma)
    clamp_min = float(cfg.clamp_min)
    clamp_max = float(cfg.clamp_max)

    if blur_sigma_px < 0.0:
        raise ValueError("blur_sigma_px must be >= 0")
    if noise_sigma < 0.0:
        raise ValueError("noise_sigma must be >= 0")
    if clamp_min < 0.0 or clamp_max > 255.0 or clamp_min > clamp_max:
        raise ValueError("clamp range must satisfy 0 <= clamp_min <= clamp_max <= 255")

    out = img.astype(np.float32, copy=False)

    if blur_sigma_px > 0.0:
        out = _gaussian_blur(out, sigma=blur_sigma_px)

    if noise_sigma > 0.0:
        if rng is None:
            raise ValueError("rng must be provided when noise_sigma > 0 for determinism")
        noise = rng.normal(0.0, noise_sigma, size=out.shape).astype(np.float32, copy=False)
        out = out + noise

    out = np.clip(out, clamp_min, clamp_max)
    out = np.rint(out)
    return out.astype(np.uint8, copy=False)
