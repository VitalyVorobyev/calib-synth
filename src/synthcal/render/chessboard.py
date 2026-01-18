"""Analytic planar chessboard rendering (no OpenCV dependency)."""

from __future__ import annotations

from typing import Any

import numpy as np

from synthcal.camera import PinholeCamera
from synthcal.core.geometry import as_se3
from synthcal.targets import ChessboardTarget


def render_chessboard_image(
    camera: PinholeCamera,
    target: ChessboardTarget,
    T_cam_target: Any,
    *,
    background: int = 128,
    supersample: int = 1,
) -> np.ndarray:
    """Render a chessboard target into a grayscale image.

    Parameters
    ----------
    camera:
        Pinhole camera model.
    target:
        Chessboard target model (plane Z=0 in target frame).
    T_cam_target:
        SE(3) transform mapping target-frame points into camera frame.
    background:
        Background grayscale intensity (0..255).
    supersample:
        Placeholder for future antialiasing. Only `1` is supported in v0.

    Returns
    -------
    np.ndarray
        Image array of shape `(H, W)` with dtype `uint8`.
    """

    if not (0 <= background <= 255):
        raise ValueError("background must be in [0, 255]")
    if supersample != 1:
        raise NotImplementedError("supersample != 1 is not implemented yet")

    width_px, height_px = camera.resolution
    H, W = int(height_px), int(width_px)

    T = as_se3(T_cam_target)
    R = T[:3, :3]
    t = T[:3, 3]

    n = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    numerator = float(np.dot(n, t))

    Kinv = np.linalg.inv(camera.K)

    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)  # (H, W)
    uu_f = uu.reshape(-1)
    vv_f = vv.reshape(-1)

    uv1 = np.stack([uu_f, vv_f, np.ones_like(uu_f)], axis=0)  # (3, N)
    xy1 = Kinv @ uv1
    xd = xy1[0] / xy1[2]
    yd = xy1[1] / xy1[2]

    xu, yu = camera.undistort_normalized(xd, yd)
    xu = np.asarray(xu, dtype=np.float64).reshape(-1)
    yu = np.asarray(yu, dtype=np.float64).reshape(-1)

    d = np.stack([xu, yu, np.ones_like(xu)], axis=0)  # (3, N)
    denom = (n.reshape(1, 3) @ d).reshape(-1)  # (N,)

    eps_denom = 1e-12
    denom_ok = np.abs(denom) > eps_denom

    t_ray = np.full_like(denom, np.nan, dtype=np.float64)
    t_ray[denom_ok] = numerator / denom[denom_ok]
    valid = denom_ok & (t_ray > 0.0)

    X_cam = d * t_ray.reshape(1, -1)  # (3, N) with NaNs for invalid rays
    X_target = R.T @ (X_cam - t.reshape(3, 1))  # (3, N)

    x = X_target[0]
    y = X_target[1]

    width_mm, height_mm = target.bounds()
    inside = (x >= 0.0) & (x < width_mm) & (y >= 0.0) & (y < height_mm)
    mask = valid & inside

    out = np.full(W * H, background, dtype=np.uint8)
    if bool(np.any(mask)):
        out[mask] = target.eval_color_xy(x[mask], y[mask])
    return out.reshape(H, W)

