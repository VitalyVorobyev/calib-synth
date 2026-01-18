"""Ground-truth projection helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from synthcal.camera import PinholeCamera
from synthcal.core.geometry import as_se3


def project_corners_px(
    camera: PinholeCamera,
    corners_xyz_target: np.ndarray,
    T_cam_target: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Project target-frame points into the image.

    Visibility (v0) is defined as:
    - point in front: `z_cam > 0`
    - projected pixel inside image bounds
    - finite pixel values
    """

    corners = np.asarray(corners_xyz_target, dtype=np.float64)
    if corners.ndim != 2 or corners.shape[1] != 3:
        raise ValueError(f"corners_xyz_target must have shape (N, 3), got {corners.shape}")

    T = as_se3(T_cam_target)
    R = T[:3, :3]
    t = T[:3, 3]

    # X_cam = R * X_target + t (row-vector form)
    X_cam = corners @ R.T + t
    z = X_cam[:, 2]

    behind = z <= 0.0
    x = np.full_like(z, np.nan, dtype=np.float64)
    y = np.full_like(z, np.nan, dtype=np.float64)

    ok = ~behind
    x[ok] = X_cam[ok, 0] / z[ok]
    y[ok] = X_cam[ok, 1] / z[ok]

    xd, yd = camera.distort_normalized(x, y)
    u, v = camera.normalized_to_pixel(xd, yd)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    corners_px = np.stack([u, v], axis=1).astype(np.float32, copy=False)

    W, H = camera.resolution
    finite = np.isfinite(u) & np.isfinite(v)
    in_bounds = (u >= 0.0) & (u < float(W)) & (v >= 0.0) & (v < float(H))
    visible = ok & finite & in_bounds
    return corners_px, visible.astype(bool, copy=False)

