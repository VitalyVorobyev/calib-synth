"""Visibility constraints for pose sampling.

All geometry is expressed in millimeters (mm).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from synthcal.camera import PinholeCamera
from synthcal.core.geometry import invert_se3
from synthcal.targets.chessboard import ChessboardTarget


def check_frame_visibility(
    cameras: dict[str, PinholeCamera],
    rig_extrinsics: dict[str, Any],
    target: ChessboardTarget,
    T_world_target: Any,
    T_base_tcp: Any,
    *,
    margin_px: int,
) -> dict[str, bool]:
    """Check whether a pose satisfies the in-view constraints per camera.

    A camera is considered "visible" when all inner corners are:
    - in front of the camera (`z_cam > 0`)
    - finite after projection
    - within image bounds with a pixel margin
    """

    if margin_px < 0:
        raise ValueError("margin_px must be >= 0")

    T_world_target = np.asarray(T_world_target, dtype=np.float64)
    T_base_tcp = np.asarray(T_base_tcp, dtype=np.float64)
    if T_world_target.shape != (4, 4) or T_base_tcp.shape != (4, 4):
        raise ValueError("T_world_target and T_base_tcp must be 4x4 matrices")

    corners_xyz = target.corners_xyz()
    margin = float(margin_px)

    out: dict[str, bool] = {}
    for cam_id, cam in cameras.items():
        T_tcp_cam = np.asarray(rig_extrinsics[cam_id], dtype=np.float64)
        if T_tcp_cam.shape != (4, 4):
            raise ValueError(f"T_tcp_cam for {cam_id} must be 4x4, got {T_tcp_cam.shape}")

        T_world_cam = T_base_tcp @ T_tcp_cam
        T_cam_target = invert_se3(T_world_cam) @ T_world_target

        R = T_cam_target[:3, :3]
        t = T_cam_target[:3, 3]
        X_cam = corners_xyz @ R.T + t
        z = X_cam[:, 2]

        ok = z > 0.0
        x = np.full_like(z, np.nan, dtype=np.float64)
        y = np.full_like(z, np.nan, dtype=np.float64)
        x[ok] = X_cam[ok, 0] / z[ok]
        y[ok] = X_cam[ok, 1] / z[ok]

        xd, yd = cam.distort_normalized(x, y)
        u, v = cam.normalized_to_pixel(xd, yd)
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

        finite = np.isfinite(u) & np.isfinite(v)
        W, H = cam.resolution
        in_bounds = (
            (u >= margin) & (u < float(W) - margin) & (v >= margin) & (v < float(H) - margin)
        )
        out[cam_id] = bool(np.all(ok & finite & in_bounds))

    return out
