"""Image effects pipeline (v0).

The pipeline is deterministic given an explicit RNG:
    blur -> noise -> clip -> quantize
"""

from __future__ import annotations

from typing import Any

import numpy as np

from synthcal.effects.config import EffectsConfig

try:  # pragma: no cover (scipy is a required dependency in this repo)
    from scipy.ndimage import gaussian_filter
except Exception as exc:  # pragma: no cover
    raise ImportError("scipy is required for gaussian blur effects") from exc


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
        out = gaussian_filter(out, sigma=blur_sigma_px, mode="nearest").astype(
            np.float32, copy=False
        )

    if noise_sigma > 0.0:
        if rng is None:
            raise ValueError("rng must be provided when noise_sigma > 0 for determinism")
        noise = rng.normal(0.0, noise_sigma, size=out.shape).astype(np.float32, copy=False)
        out = out + noise

    out = np.clip(out, clamp_min, clamp_max)
    out = np.rint(out)
    return out.astype(np.uint8, copy=False)
