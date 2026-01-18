"""Dataset generation (v0).

This module currently supports generating:
- a rendered chessboard image per frame/camera
- ground-truth inner corner projections + visibility mask per frame/camera
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from synthcal import __version__
from synthcal.camera import PinholeCamera
from synthcal.core.geometry import invert_se3
from synthcal.core.seeding import derive_rng
from synthcal.effects.pipeline import apply_effects
from synthcal.io.config import SynthCalConfig, save_config
from synthcal.io.manifest import (
    ManifestCamera,
    ManifestGenerator,
    ManifestLaser,
    ManifestLayout,
    ManifestPaths,
    SynthCalManifest,
    save_manifest,
    utc_now_iso8601,
)
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


def _mat44(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {arr.shape}")
    return arr


def _default_T_world_target(target: ChessboardTarget) -> np.ndarray:
    """Default target pose that is fronto-parallel and centered in view."""

    width_mm, height_mm = target.bounds()
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.array([-width_mm / 2.0, -height_mm / 2.0, 1000.0], dtype=np.float64)
    return T


def _yaml_dump(data: object) -> str:
    import yaml

    return yaml.safe_dump(
        data,
        sort_keys=False,
        indent=2,
        default_flow_style=False,
    )


def _write_rig_files(cfg: SynthCalConfig, out_dir: Path) -> None:
    rig_dir = out_dir / "rig"
    cams_dir = rig_dir / "cameras"
    cams_dir.mkdir(parents=True, exist_ok=True)

    rig_yaml = {
        "version": 1,
        "units": {"length": "mm"},
        "cameras": [
            {
                "name": cam.name,
                "intrinsics_yaml": f"cameras/{cam.name}.yaml",
                "T_tcp_cam": [list(row) for row in cam.T_tcp_cam],
            }
            for cam in cfg.rig.cameras
        ],
        "notes": "Placeholder rig file (v0).",
    }
    (rig_dir / "rig.yaml").write_text(_yaml_dump(rig_yaml), encoding="utf-8")

    for cam in cfg.rig.cameras:
        (cams_dir / f"{cam.name}.yaml").write_text(
            _yaml_dump(
                {
                    "version": 1,
                    "name": cam.name,
                    "image_size_px": [cam.image_size_px[0], cam.image_size_px[1]],
                    "K": [list(row) for row in cam.K],
                    "dist": list(cam.dist),
                    "notes": "Placeholder intrinsics file (v0).",
                }
            ),
            encoding="utf-8",
        )


def generate_dataset(cfg: SynthCalConfig, out_dir: str | Path) -> None:
    """Generate a dataset on disk."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist the normalized config used for generation.
    save_config(cfg, out_dir / "config.yaml")
    _write_rig_files(cfg, out_dir)

    # Build camera/target models.
    cols, rows = cfg.chessboard.inner_corners
    target = ChessboardTarget(
        inner_rows=rows, inner_cols=cols, square_size_mm=cfg.chessboard.square_size_mm
    )
    corners_xyz = target.corners_xyz()

    if cfg.scene is not None:
        T_world_target = _mat44(cfg.scene.T_world_target)
    else:
        T_world_target = _default_T_world_target(target)

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    camera_models: dict[str, PinholeCamera] = {}
    rig_T_tcp_cam: dict[str, np.ndarray] = {}
    for cam_cfg in cfg.rig.cameras:
        camera_models[cam_cfg.name] = PinholeCamera(
            resolution=cam_cfg.image_size_px,
            K=np.asarray(cam_cfg.K, dtype=np.float64),
            dist=np.asarray(cam_cfg.dist, dtype=np.float64),
        )
        rig_T_tcp_cam[cam_cfg.name] = _mat44(cam_cfg.T_tcp_cam)

    laser_plane_tcp = None
    if cfg.laser is not None and cfg.laser.enabled:
        laser_plane_tcp = normalize_plane(np.asarray(cfg.laser.plane_in_tcp, dtype=np.float64))
    board_plane_target = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)

    for frame_index in range(cfg.dataset.num_frames):
        frame_dir = frames_dir / f"frame_{frame_index:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        if cfg.scenario is None:
            T_base_tcp = np.eye(4, dtype=np.float64)
        else:
            reference_cam = cfg.rig.cameras[0].name
            T_base_tcp, _vis = sample_valid_T_base_tcp(
                global_seed=cfg.seed,
                frame_id=frame_index,
                scenario=cfg.scenario,
                cameras=camera_models,
                rig_extrinsics=rig_T_tcp_cam,
                target=target,
                T_world_target=T_world_target,
                reference_camera=reference_cam,
            )
        np.save(frame_dir / "T_base_tcp.npy", T_base_tcp)

        for cam_cfg in cfg.rig.cameras:
            cam = camera_models[cam_cfg.name]

            T_tcp_cam = rig_T_tcp_cam[cam_cfg.name]
            T_world_cam = T_base_tcp @ T_tcp_cam
            T_cam_target = invert_se3(T_world_cam) @ T_world_target

            img = render_chessboard_image(cam, target, T_cam_target, supersample=2)
            rng_target = derive_rng(cfg.seed, frame_index, cam_cfg.name, "target")
            img = apply_effects(img, cfg.effects, rng_target)
            Image.fromarray(img, mode="L").save(frame_dir / f"{cam_cfg.name}_target.png")

            corners_px, visible = project_corners_px(cam, corners_xyz, T_cam_target)
            np.save(
                frame_dir / f"{cam_cfg.name}_corners_px.npy",
                corners_px.astype(np.float32, copy=False),
            )
            np.save(
                frame_dir / f"{cam_cfg.name}_corners_visible.npy", visible.astype(bool, copy=False)
            )

            if laser_plane_tcp is not None:
                T_cam_tcp = invert_se3(T_tcp_cam)
                plane_cam = transform_plane(T_cam_tcp, laser_plane_tcp)
                T_target_cam = invert_se3(T_cam_target)
                plane_target = transform_plane(T_target_cam, plane_cam)

                stripe_px: np.ndarray
                stripe_vis: np.ndarray
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

                rng_stripe = derive_rng(cfg.seed, frame_index, cam_cfg.name, "stripe")
                stripe_img = apply_effects(stripe_img, cfg.effects, rng_stripe)
                Image.fromarray(stripe_img, mode="L").save(frame_dir / f"{cam_cfg.name}_stripe.png")
                np.save(
                    frame_dir / f"{cam_cfg.name}_stripe_centerline_px.npy",
                    stripe_px.astype(np.float32, copy=False),
                )
                np.save(
                    frame_dir / f"{cam_cfg.name}_stripe_centerline_visible.npy",
                    stripe_vis.astype(bool, copy=False),
                )

    # Write a v1 manifest that lists only outputs produced in v0.
    include_laser = cfg.laser is not None and cfg.laser.enabled
    manifest = SynthCalManifest(
        manifest_version=1,
        created_utc=utc_now_iso8601(),
        generator=ManifestGenerator(name="synthcal", version=__version__),
        seed=cfg.seed,
        units={"length": "mm"},
        dataset={"name": cfg.dataset.name, "num_frames": cfg.dataset.num_frames},
        laser=(
            None
            if not include_laser
            else ManifestLaser(
                enabled=True,
                plane_in_tcp=tuple(float(v) for v in cfg.laser.plane_in_tcp),
                stripe_width_px=int(cfg.laser.stripe_width_px),
                stripe_intensity=int(cfg.laser.stripe_intensity),
            )
        ),
        paths=ManifestPaths(
            config_yaml="config.yaml",
            manifest_yaml="manifest.yaml",
            rig_yaml="rig/rig.yaml",
            cameras_dir="rig/cameras",
            frames_dir="frames",
        ),
        cameras=tuple(
            ManifestCamera(
                name=c.name,
                intrinsics_yaml=f"rig/cameras/{c.name}.yaml",
                image_size_px=c.image_size_px,
            )
            for c in cfg.rig.cameras
        ),
        layout=ManifestLayout.v1_default(include_laser=include_laser),
    )
    save_manifest(manifest, out_dir / "manifest.yaml")
