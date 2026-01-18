"""Laser stripe rasterization (v0).

This module intentionally implements a simple, deterministic renderer suitable
for preview and small/moderate datasets.

All geometry upstream is expressed in millimeters (mm). This renderer operates
purely in pixel space.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from synthcal.camera import PinholeCamera


def render_stripe_image(
    camera: PinholeCamera,
    centerline_px: Any,
    centerline_visible: Any,
    *,
    width_px: float,
    intensity: int = 255,
    background: int = 0,
) -> np.ndarray:
    """Render a stripe-only grayscale image by splatting a Gaussian kernel.

    Parameters
    ----------
    camera:
        Pinhole camera model.
    centerline_px:
        Array-like of shape `(M, 2)` containing projected `(u, v)` pixel coordinates.
    centerline_visible:
        Boolean mask of shape `(M,)` indicating which samples should contribute.
    width_px:
        Gaussian sigma in pixels.
    intensity:
        Peak intensity at the stripe center (0..255).
    background:
        Background grayscale intensity (0..255).

    Returns
    -------
    np.ndarray
        Image array of shape `(H, W)` with dtype `uint8`.
    """

    if width_px <= 0.0:
        raise ValueError("width_px must be > 0")
    if not (0 <= intensity <= 255):
        raise ValueError("intensity must be in [0, 255]")
    if not (0 <= background <= 255):
        raise ValueError("background must be in [0, 255]")

    pts = np.asarray(centerline_px, dtype=np.float64)
    vis = np.asarray(centerline_visible, dtype=bool)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"centerline_px must have shape (M, 2), got {pts.shape}")
    if vis.shape != (pts.shape[0],):
        raise ValueError(f"centerline_visible must have shape ({pts.shape[0]},), got {vis.shape}")

    width_px = float(width_px)
    sigma2 = width_px * width_px
    r = int(math.ceil(3.0 * width_px))

    W, H = camera.resolution
    img = np.full((int(H), int(W)), float(background), dtype=np.float32)

    if not bool(np.any(vis)):
        return img.astype(np.uint8, copy=False)

    peak = float(intensity)
    base = float(background)
    for u_f, v_f in pts[vis]:
        if not (np.isfinite(u_f) and np.isfinite(v_f)):
            continue

        u0 = int(np.round(u_f))
        v0 = int(np.round(v_f))

        u_min = max(u0 - r, 0)
        u_max = min(u0 + r, int(W) - 1)
        v_min = max(v0 - r, 0)
        v_max = min(v0 + r, int(H) - 1)

        for v in range(v_min, v_max + 1):
            dv = v - v0
            for u in range(u_min, u_max + 1):
                du = u - u0
                w = math.exp(-((du * du + dv * dv) / (2.0 * sigma2)))
                val = base + (peak - base) * float(w)
                if val > img[v, u]:
                    img[v, u] = val

    return np.clip(img, 0.0, 255.0).astype(np.uint8, copy=False)
