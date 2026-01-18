"""Command-line interface.

Entry point: `python -m synthcal ...`
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from synthcal.io.config import SynthCalConfig, load_config, save_config
from synthcal.io.generate import generate_dataset


def _cmd_init_config(path: Path) -> int:
    cfg = SynthCalConfig.example()
    save_config(cfg, path)
    return 0


def _cmd_generate(config_path: Path, out_dir: Path) -> int:
    cfg = load_config(config_path)
    generate_dataset(cfg, out_dir)
    return 0


def _cmd_preview(
    config_path: Path,
    *,
    frame_index: int,
    camera_name: str | None,
    all_cams: bool,
    show_stripe: bool,
    no_effects: bool,
) -> int:
    import numpy as np

    from synthcal.camera import PinholeCamera
    from synthcal.core.geometry import invert_se3
    from synthcal.core.seeding import derive_rng
    from synthcal.effects.pipeline import apply_effects
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

    cfg = load_config(config_path)
    if not (0 <= frame_index < cfg.dataset.num_frames):
        raise ValueError(f"--frame must be in [0, {cfg.dataset.num_frames - 1}]")

    cam_cfgs = list(cfg.rig.cameras)
    cam_cfg = None
    if not all_cams:
        if camera_name is None:
            cam_cfg = cfg.rig.cameras[0]
        else:
            for c in cfg.rig.cameras:
                if c.name == camera_name:
                    cam_cfg = c
                    break
            if cam_cfg is None:
                available = [c.name for c in cfg.rig.cameras]
                raise ValueError(f"Unknown camera {camera_name!r}; available: {available}")

    cols, rows = cfg.chessboard.inner_corners
    target = ChessboardTarget(
        inner_rows=rows, inner_cols=cols, square_size_mm=cfg.chessboard.square_size_mm
    )
    corners_xyz = target.corners_xyz()

    if cfg.scene is not None:
        T_world_target = np.asarray(cfg.scene.T_world_target, dtype=np.float64)
    else:
        width_mm, height_mm = target.bounds()
        T_world_target = np.eye(4, dtype=np.float64)
        T_world_target[:3, 3] = np.array(
            [-width_mm / 2.0, -height_mm / 2.0, 1000.0], dtype=np.float64
        )

    cameras: dict[str, PinholeCamera] = {}
    rig_T_tcp_cam: dict[str, np.ndarray] = {}
    for c in cfg.rig.cameras:
        cameras[c.name] = PinholeCamera(
            resolution=c.image_size_px,
            K=np.asarray(c.K, dtype=np.float64),
            dist=np.asarray(c.dist, dtype=np.float64),
        )
        rig_T_tcp_cam[c.name] = np.asarray(c.T_tcp_cam, dtype=np.float64)

    vis_by_cam: dict[str, bool] = {}
    if cfg.scenario is None:
        T_base_tcp = np.eye(4, dtype=np.float64)
    else:
        reference_cam = cfg.rig.cameras[0].name
        T_base_tcp, vis_by_cam = sample_valid_T_base_tcp(
            global_seed=cfg.seed,
            frame_id=frame_index,
            scenario=cfg.scenario,
            cameras=cameras,
            rig_extrinsics=rig_T_tcp_cam,
            target=target,
            T_world_target=T_world_target,
            reference_camera=reference_cam,
        )
        vis_str = ", ".join(f"{k}={int(v)}" for k, v in sorted(vis_by_cam.items()))
        logging.info("scenario visibility: %s", vis_str)

    import matplotlib.pyplot as plt

    show_laser = cfg.laser is not None and cfg.laser.enabled
    show_stripe_effective = bool(show_laser and (show_stripe or not all_cams))

    def _render_target(
        cam_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, PinholeCamera, np.ndarray]:
        cam = cameras[cam_name]
        T_tcp_cam = rig_T_tcp_cam[cam_name]
        T_world_cam = T_base_tcp @ T_tcp_cam
        T_cam_target = invert_se3(T_world_cam) @ T_world_target

        img = render_chessboard_image(cam, target, T_cam_target)
        corners_px, visible = project_corners_px(cam, corners_xyz, T_cam_target)
        if not no_effects:
            rng_target = derive_rng(cfg.seed, frame_index, cam_name, "target")
            img = apply_effects(img, cfg.effects, rng_target)
        return img, corners_px, visible, cam, T_cam_target

    def _render_stripe(
        cam_name: str, cam: PinholeCamera, T_cam_target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        plane_tcp = normalize_plane(np.asarray(cfg.laser.plane_in_tcp, dtype=np.float64))
        board_plane_target = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)

        T_tcp_cam = rig_T_tcp_cam[cam_name]
        T_cam_tcp = invert_se3(T_tcp_cam)
        plane_cam = transform_plane(T_cam_tcp, plane_tcp)
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

        if not no_effects:
            rng_stripe = derive_rng(cfg.seed, frame_index, cam_name, "stripe")
            stripe_img = apply_effects(stripe_img, cfg.effects, rng_stripe)

        return stripe_img, stripe_px, stripe_vis

    def _plot_target(ax: Any, cam_name: str) -> np.ndarray:
        img, corners_px, visible, cam, _T_cam_target = _render_target(cam_name)

        ok = vis_by_cam.get(cam_name, True)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{cam_name} ok={int(ok)}")
        if bool(np.any(visible)):
            ax.scatter(corners_px[visible, 0], corners_px[visible, 1], s=12, c="lime", marker="o")
        if bool(np.any(~visible)):
            ax.scatter(corners_px[~visible, 0], corners_px[~visible, 1], s=12, c="red", marker="x")
        ax.set_xlim([-0.5, cam.resolution[0] - 0.5])
        ax.set_ylim([cam.resolution[1] - 0.5, -0.5])
        return _T_cam_target

    def _plot_stripe(ax: Any, cam_name: str, cam: PinholeCamera, T_cam_target: np.ndarray) -> None:
        stripe_img, stripe_px, stripe_vis = _render_stripe(cam_name, cam, T_cam_target)
        ok = vis_by_cam.get(cam_name, True)
        ax.imshow(stripe_img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{cam_name} stripe ok={int(ok)}")
        if stripe_px.size:
            finite = np.isfinite(stripe_px[:, 0]) & np.isfinite(stripe_px[:, 1])
            vis = stripe_vis & finite
            inv = (~stripe_vis) & finite
            if bool(np.any(vis)):
                ax.scatter(stripe_px[vis, 0], stripe_px[vis, 1], s=6, c="cyan", marker=".")
            if bool(np.any(inv)):
                ax.scatter(stripe_px[inv, 0], stripe_px[inv, 1], s=10, c="red", marker="x")
        ax.set_xlim([-0.5, cam.resolution[0] - 0.5])
        ax.set_ylim([cam.resolution[1] - 0.5, -0.5])

    if all_cams:
        import math

        cam_names = [c.name for c in cam_cfgs]
        n = len(cam_names)
        cols = min(3, n)
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes_arr = np.atleast_1d(axes).reshape(-1)
        T_cam_targets: dict[str, np.ndarray] = {}
        for i, name in enumerate(cam_names):
            T_cam_targets[name] = _plot_target(axes_arr[i], name)
        for ax in axes_arr[n:]:
            ax.axis("off")
        fig.suptitle(f"frame={frame_index} targets")

        if show_stripe_effective:
            fig2, axes2 = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes2_arr = np.atleast_1d(axes2).reshape(-1)
            for i, name in enumerate(cam_names):
                _plot_stripe(
                    axes2_arr[i],
                    name,
                    cameras[name],
                    T_cam_targets[name],
                )
            for ax in axes2_arr[n:]:
                ax.axis("off")
            fig2.suptitle(f"frame={frame_index} stripes")
    else:
        assert cam_cfg is not None
        cam_name = cam_cfg.name
        img, corners_px, visible, cam, T_cam_target = _render_target(cam_name)

        if show_stripe_effective:
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
        else:
            fig, ax0 = plt.subplots()
            ax1 = None

        ax0.imshow(img, cmap="gray", vmin=0, vmax=255)
        ok = vis_by_cam.get(cam_name, True)
        ax0.set_title(f"target frame={frame_index} cam={cam_name} ok={int(ok)}")

        if bool(np.any(visible)):
            ax0.scatter(corners_px[visible, 0], corners_px[visible, 1], s=12, c="lime", marker="o")
        if bool(np.any(~visible)):
            ax0.scatter(corners_px[~visible, 0], corners_px[~visible, 1], s=12, c="red", marker="x")

        ax0.set_xlim([-0.5, cam.resolution[0] - 0.5])
        ax0.set_ylim([cam.resolution[1] - 0.5, -0.5])

        if show_stripe_effective and ax1 is not None:
            _plot_stripe(ax1, cam_name, cam, T_cam_target)

    plt.show()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthcal", description="Synthetic calibration dataset generator"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-config", help="Write an example config.yaml")
    p_init.add_argument("path", type=Path, help="Output path for config.yaml")

    p_gen = sub.add_parser("generate", help="Generate dataset (renders chessboard + GT corners)")
    p_gen.add_argument("config_yaml", type=Path, help="Input config.yaml")
    p_gen.add_argument("out_dir", type=Path, help="Output dataset directory")

    p_prev = sub.add_parser(
        "preview", help="Render a single frame/camera and show a preview window"
    )
    p_prev.add_argument("config_yaml", type=Path, help="Input config.yaml")
    p_prev.add_argument("--frame", type=int, default=0, help="Frame index (default: 0)")
    p_prev.add_argument(
        "--cam",
        type=str,
        default=None,
        help="Camera name (default: first camera in config)",
    )
    p_prev.add_argument(
        "--all-cams",
        action="store_true",
        help="Show a grid preview for all cameras",
    )
    p_prev.add_argument(
        "--show-stripe",
        action="store_true",
        help="When used with --all-cams, also show a stripe grid (if laser is enabled)",
    )
    p_prev.add_argument(
        "--no-effects",
        action="store_true",
        help="Bypass effects (blur/noise/quantize) and show the raw render",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    try:
        if args.command == "init-config":
            return _cmd_init_config(args.path)
        if args.command == "generate":
            return _cmd_generate(args.config_yaml, args.out_dir)
        if args.command == "preview":
            return _cmd_preview(
                args.config_yaml,
                frame_index=args.frame,
                camera_name=args.cam,
                all_cams=args.all_cams,
                show_stripe=args.show_stripe,
                no_effects=args.no_effects,
            )
        raise AssertionError(f"Unhandled command: {args.command}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
