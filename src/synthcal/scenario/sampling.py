"""Pose sampling utilities for scenario generation.

All geometry is expressed in millimeters (mm).

Sampling strategy (v0)
----------------------
For a reference camera (first camera in the rig), sample a camera pose around a point on the
target plane, then derive a TCP pose using the known `T_tcp_cam` extrinsic.

Rotation parameters:
- yaw: azimuth around the target Z axis
- tilt: polar angle away from the target Z axis
- roll: rotation around the camera viewing axis

Translation parameters:
- distance: radial distance from the sampled look-at point
- xy_offset_frac: look-at point offset in the target plane as fractions of board size

The output is `T_base_tcp` (world==base in v0).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from synthcal.camera import PinholeCamera
from synthcal.core.geometry import invert_se3
from synthcal.core.seeding import derive_rng
from synthcal.scenario.config import ScenarioConfig
from synthcal.scenario.constraints import check_frame_visibility
from synthcal.targets.chessboard import ChessboardTarget


def _deg2rad(deg: float) -> float:
    return float(deg) * math.pi / 180.0


def _normalize(v: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n


def _axis_angle_to_R(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize(axis)
    x, y, z = (float(axis[0]), float(axis[1]), float(axis[2]))
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _sample_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    lo_f, hi_f = float(lo), float(hi)
    if hi_f < lo_f:
        raise ValueError("Invalid range: max < min")
    return float(rng.uniform(lo_f, hi_f))


def sample_T_base_tcp(
    rng: np.random.Generator,
    *,
    target: ChessboardTarget,
    T_world_target: Any,
    T_tcp_cam_ref: Any,
    scenario: ScenarioConfig,
) -> np.ndarray:
    """Sample a single `T_base_tcp` pose."""

    width_mm, height_mm = target.bounds()
    center = np.array([width_mm / 2.0, height_mm / 2.0, 0.0], dtype=np.float64)

    d = _sample_uniform(rng, scenario.distance_mm.min, scenario.distance_mm.max)
    tilt = _deg2rad(_sample_uniform(rng, scenario.tilt_deg.min, scenario.tilt_deg.max))
    yaw = _deg2rad(_sample_uniform(rng, scenario.yaw_deg.min, scenario.yaw_deg.max))
    roll = _deg2rad(_sample_uniform(rng, scenario.roll_deg.min, scenario.roll_deg.max))

    frac_x = _sample_uniform(rng, scenario.xy_offset_frac.min, scenario.xy_offset_frac.max)
    frac_y = _sample_uniform(rng, scenario.xy_offset_frac.min, scenario.xy_offset_frac.max)
    look_at = center + np.array([frac_x * width_mm, frac_y * height_mm, 0.0], dtype=np.float64)

    z_cam_in_target = np.array(
        [math.sin(tilt) * math.cos(yaw), math.sin(tilt) * math.sin(yaw), math.cos(tilt)],
        dtype=np.float64,
    )
    z_cam_in_target = _normalize(z_cam_in_target)

    cam_pos_in_target = look_at - float(d) * z_cam_in_target

    # Build camera axes in target coordinates.
    y_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_cam_in_target = np.cross(y_ref, z_cam_in_target)
    if float(np.linalg.norm(x_cam_in_target)) < 1e-9:
        x_cam_in_target = np.cross(np.array([1.0, 0.0, 0.0], dtype=np.float64), z_cam_in_target)
    x_cam_in_target = _normalize(x_cam_in_target)
    y_cam_in_target = np.cross(z_cam_in_target, x_cam_in_target)
    y_cam_in_target = _normalize(y_cam_in_target)

    if abs(roll) > 0.0:
        R_roll = _axis_angle_to_R(z_cam_in_target, roll)
        x_cam_in_target = R_roll @ x_cam_in_target
        y_cam_in_target = R_roll @ y_cam_in_target

    R_target_cam = np.stack([x_cam_in_target, y_cam_in_target, z_cam_in_target], axis=1)
    T_target_cam = np.eye(4, dtype=np.float64)
    T_target_cam[:3, :3] = R_target_cam
    T_target_cam[:3, 3] = cam_pos_in_target

    T_world_target = np.asarray(T_world_target, dtype=np.float64)
    if T_world_target.shape != (4, 4):
        raise ValueError(f"T_world_target must be 4x4, got {T_world_target.shape}")
    T_world_cam = T_world_target @ T_target_cam

    T_tcp_cam_ref = np.asarray(T_tcp_cam_ref, dtype=np.float64)
    if T_tcp_cam_ref.shape != (4, 4):
        raise ValueError(f"T_tcp_cam_ref must be 4x4, got {T_tcp_cam_ref.shape}")
    T_world_tcp = T_world_cam @ invert_se3(T_tcp_cam_ref)
    return T_world_tcp


def sample_valid_T_base_tcp(
    *,
    global_seed: int,
    frame_id: int,
    scenario: ScenarioConfig,
    cameras: dict[str, PinholeCamera],
    rig_extrinsics: dict[str, Any],
    target: ChessboardTarget,
    T_world_target: Any,
    reference_camera: str,
) -> tuple[np.ndarray, dict[str, bool]]:
    """Sample a `T_base_tcp` pose that satisfies the scenario constraints."""

    rng = derive_rng(global_seed, frame_id, "__scenario__", "pose")

    in_view = scenario.in_view
    if in_view.require_all_cameras:
        min_visible = len(cameras)
    else:
        min_visible = int(in_view.min_cameras_visible)
        if min_visible <= 0:
            raise ValueError("scenario.in_view.min_cameras_visible must be > 0")
        if min_visible > len(cameras):
            raise ValueError("scenario.in_view.min_cameras_visible cannot exceed number of cameras")

    T_tcp_cam_ref = rig_extrinsics[reference_camera]
    last_vis: dict[str, bool] | None = None
    for _attempt in range(int(scenario.max_attempts_per_frame)):
        T_base_tcp = sample_T_base_tcp(
            rng,
            target=target,
            T_world_target=T_world_target,
            T_tcp_cam_ref=T_tcp_cam_ref,
            scenario=scenario,
        )
        vis = check_frame_visibility(
            cameras,
            rig_extrinsics,
            target,
            T_world_target,
            T_base_tcp,
            margin_px=in_view.margin_px,
        )
        last_vis = vis
        if sum(bool(v) for v in vis.values()) >= min_visible:
            return T_base_tcp, vis

    vis_str = "(no visibility results)"
    if last_vis is not None:
        vis_str = ", ".join(f"{k}={int(v)}" for k, v in sorted(last_vis.items()))
    raise ValueError(
        "Failed to sample a valid pose for frame "
        f"{frame_id} after {scenario.max_attempts_per_frame} attempts. "
        f"Last visibility: {vis_str}. "
        "Try increasing max_attempts_per_frame, reducing tilt/offset, increasing distance, "
        "reducing margin_px, or lowering the required number of visible cameras."
    )
