"""Analytic planar chessboard rendering (no OpenCV dependency).

The renderer models the chessboard as a planar texture in the target frame (Z=0) and uses
ray–plane intersections to compute, for each image sample, the corresponding `(x_mm, y_mm)`
coordinate on the target plane.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from synthcal.camera import PinholeCamera
from synthcal.core.geometry import as_se3
from synthcal.targets import ChessboardTarget


def _render_chessboard_samples(
    camera: PinholeCamera,
    target: ChessboardTarget,
    T_cam_target: Any,
    *,
    background: int,
    u_px: np.ndarray,
    v_px: np.ndarray,
) -> np.ndarray:
    """Render a chessboard evaluated at an explicit `(u_px, v_px)` sample grid.

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
    u_px, v_px:
        1D arrays of sample coordinates in pixel units. The output image will have shape
        `(len(v_px), len(u_px))`.
    """

    width_samples_px = int(u_px.shape[0])
    height_samples_px = int(v_px.shape[0])

    T = as_se3(T_cam_target)
    R = T[:3, :3]
    t = T[:3, 3]

    n = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    numerator = float(np.dot(n, t))

    Kinv = np.linalg.inv(camera.K)

    uu, vv = np.meshgrid(u_px.astype(np.float64, copy=False), v_px.astype(np.float64, copy=False))
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

    out = np.full(width_samples_px * height_samples_px, background, dtype=np.uint8)
    if bool(np.any(mask)):
        out[mask] = target.eval_color_xy(x[mask], y[mask])
    return out.reshape(height_samples_px, width_samples_px)


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
        Supersampling factor for antialiasing. When `supersample > 1`, the renderer evaluates the
        chessboard on a `supersample x supersample` grid within each pixel and averages the results.

    Returns
    -------
    np.ndarray
        Image array of shape `(H, W)` with dtype `uint8`.
    """

    if not (0 <= background <= 255):
        raise ValueError("background must be in [0, 255]")
    if isinstance(supersample, bool) or not isinstance(supersample, int):
        raise TypeError("supersample must be an integer")
    if supersample <= 0:
        raise ValueError("supersample must be > 0")

    width_px, height_px = camera.resolution
    H, W = int(height_px), int(width_px)

    if supersample == 1:
        u = np.arange(W, dtype=np.float64)
        v = np.arange(H, dtype=np.float64)
        return _render_chessboard_samples(
            camera,
            target,
            T_cam_target,
            background=background,
            u_px=u,
            v_px=v,
        )

    # Antialiasing by sampling the pixel area: evaluate on a dense sample grid and average in
    # `supersample x supersample` blocks.
    S = int(supersample)
    u_hi = (np.arange(W * S, dtype=np.float64) + 0.5) / float(S) - 0.5
    v_hi = (np.arange(H * S, dtype=np.float64) + 0.5) / float(S) - 0.5
    img_hi = _render_chessboard_samples(
        camera,
        target,
        T_cam_target,
        background=background,
        u_px=u_hi,
        v_px=v_hi,
    ).astype(np.float32, copy=False)

    img_lo = img_hi.reshape(H, S, W, S).mean(axis=(1, 3))
    img_lo = np.clip(np.rint(img_lo), 0.0, 255.0)
    return img_lo.astype(np.uint8, copy=False)
