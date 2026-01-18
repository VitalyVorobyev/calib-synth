"""Public API for synthcal.

The functions in this module are the supported, stable entry points for using synthcal as a
library. Internal modules may change without notice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from synthcal.camera import PinholeCamera
from synthcal.core.geometry import invert_se3
from synthcal.core.seeding import derive_rng
from synthcal.effects.pipeline import apply_effects
from synthcal.io.config import SynthCalConfig, load_config
from synthcal.io.generate import generate_dataset as _generate_dataset_from_config
from synthcal.laser import (
    intersect_planes_to_line,
    normalize_plane,
    sample_line_points,
    transform_plane,
)
from synthcal.render.chessboard import render_chessboard_image
from synthcal.render.gt import project_corners_px
from synthcal.render.stripe import render_stripe_image
from synthcal.scenario.sampling import sample_valid_T_base_tcp
from synthcal.targets.chessboard import ChessboardTarget


def generate_dataset(config_path: str | Path, out_dir: str | Path) -> Path:
    """Generate a dataset from a config file on disk."""

    cfg = load_config(config_path)
    return generate_dataset_from_config(cfg, out_dir)


def generate_dataset_from_config(config_obj: SynthCalConfig, out_dir: str | Path) -> Path:
    """Generate a dataset from an in-memory config object."""

    out_path = Path(out_dir)
    _generate_dataset_from_config(config_obj, out_path)
    return out_path


def _default_T_world_target(target: ChessboardTarget) -> np.ndarray:
    width_mm, height_mm = target.bounds()
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.array([-width_mm / 2.0, -height_mm / 2.0, 1000.0], dtype=np.float64)
    return T


def render_frame_preview(config_obj: SynthCalConfig, frame_id: int) -> dict[str, Any]:
    """Render one frame worth of images and geometric ground truth.

    Returns a nested dict:
    - `T_base_tcp`: float64 (4,4)
    - `visibility`: dict(cam_name -> bool) (present only when scenario is enabled)
    - `cameras[cam_name]`: per-camera images/GT arrays
    """

    cfg = config_obj
    if not (0 <= frame_id < cfg.dataset.num_frames):
        raise ValueError(f"frame_id must be in [0, {cfg.dataset.num_frames - 1}]")

    cols, rows = cfg.chessboard.inner_corners
    target = ChessboardTarget(
        inner_rows=rows, inner_cols=cols, square_size_mm=cfg.chessboard.square_size_mm
    )
    corners_xyz = target.corners_xyz()

    if cfg.scene is not None:
        T_world_target = np.asarray(cfg.scene.T_world_target, dtype=np.float64)
    else:
        T_world_target = _default_T_world_target(target)

    camera_models: dict[str, PinholeCamera] = {}
    rig_T_tcp_cam: dict[str, np.ndarray] = {}
    for cam_cfg in cfg.rig.cameras:
        camera_models[cam_cfg.name] = PinholeCamera(
            resolution=cam_cfg.image_size_px,
            K=np.asarray(cam_cfg.K, dtype=np.float64),
            dist=np.asarray(cam_cfg.dist, dtype=np.float64),
        )
        rig_T_tcp_cam[cam_cfg.name] = np.asarray(cam_cfg.T_tcp_cam, dtype=np.float64)

    vis_by_cam: dict[str, bool] | None = None
    if cfg.scenario is None:
        T_base_tcp = np.eye(4, dtype=np.float64)
    else:
        reference_cam = cfg.rig.cameras[0].name
        T_base_tcp, vis = sample_valid_T_base_tcp(
            global_seed=cfg.seed,
            frame_id=frame_id,
            scenario=cfg.scenario,
            cameras=camera_models,
            rig_extrinsics=rig_T_tcp_cam,
            target=target,
            T_world_target=T_world_target,
            reference_camera=reference_cam,
        )
        vis_by_cam = vis

    laser_enabled = cfg.laser is not None and cfg.laser.enabled
    laser_plane_tcp = None
    if laser_enabled:
        laser_plane_tcp = normalize_plane(np.asarray(cfg.laser.plane_in_tcp, dtype=np.float64))
    board_plane_target = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)

    out: dict[str, Any] = {
        "frame_index": frame_id,
        "T_base_tcp": T_base_tcp,
        "cameras": {},
    }
    if vis_by_cam is not None:
        out["visibility"] = dict(vis_by_cam)

    for cam_cfg in cfg.rig.cameras:
        cam = camera_models[cam_cfg.name]
        T_tcp_cam = rig_T_tcp_cam[cam_cfg.name]

        T_world_cam = T_base_tcp @ T_tcp_cam
        T_cam_target = invert_se3(T_world_cam) @ T_world_target

        img_target = render_chessboard_image(cam, target, T_cam_target, supersample=2)
        rng_target = derive_rng(cfg.seed, frame_id, cam_cfg.name, "target")
        img_target = apply_effects(img_target, cfg.effects, rng_target)

        corners_px, corners_visible = project_corners_px(cam, corners_xyz, T_cam_target)

        cam_out: dict[str, Any] = {
            "T_cam_target": T_cam_target,
            "target_image_u8": img_target,
            "corners_px": corners_px,
            "corners_visible": corners_visible,
        }

        if laser_plane_tcp is not None:
            T_cam_tcp = invert_se3(T_tcp_cam)
            plane_cam = transform_plane(T_cam_tcp, laser_plane_tcp)
            T_target_cam = invert_se3(T_cam_target)
            plane_target = transform_plane(T_target_cam, plane_cam)

            try:
                p0, v = intersect_planes_to_line(plane_target, board_plane_target, eps=1e-12)
            except ValueError:
                stripe_px = np.zeros((0, 2), dtype=np.float32)
                stripe_vis = np.zeros((0,), dtype=bool)
                stripe_img = np.zeros((cam.resolution[1], cam.resolution[0]), dtype=np.uint8)
            else:
                width_mm, height_mm = target.bounds()
                L = float(max(width_mm, height_mm) * 4.0)
                num = int(max(cam.resolution) * 2)
                pts_target = sample_line_points(p0, v, -L, L, num)
                stripe_px, stripe_vis = project_corners_px(cam, pts_target, T_cam_target)
                stripe_img = render_stripe_image(
                    cam,
                    stripe_px,
                    stripe_vis,
                    width_px=float(cfg.laser.stripe_width_px),
                    intensity=int(cfg.laser.stripe_intensity),
                    background=0,
                )

            rng_stripe = derive_rng(cfg.seed, frame_id, cam_cfg.name, "stripe")
            stripe_img = apply_effects(stripe_img, cfg.effects, rng_stripe)

            cam_out.update(
                {
                    "stripe_image_u8": stripe_img,
                    "stripe_centerline_px": stripe_px,
                    "stripe_centerline_visible": stripe_vis,
                }
            )

        out["cameras"][cam_cfg.name] = cam_out

    return out
