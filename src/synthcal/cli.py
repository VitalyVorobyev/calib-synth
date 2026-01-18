"""Command-line interface.

Entry point: `python -m synthcal ...`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def _cmd_preview(config_path: Path, *, frame_index: int, camera_name: str | None) -> int:
    import numpy as np

    from synthcal.camera import PinholeCamera
    from synthcal.core.geometry import invert_se3
    from synthcal.render.chessboard import render_chessboard_image
    from synthcal.render.gt import project_corners_px
    from synthcal.targets.chessboard import ChessboardTarget

    cfg = load_config(config_path)
    if not (0 <= frame_index < cfg.dataset.num_frames):
        raise ValueError(f"--frame must be in [0, {cfg.dataset.num_frames - 1}]")

    cam_cfg = None
    if camera_name is None:
        cam_cfg = cfg.rig.cameras[0]
    else:
        for c in cfg.rig.cameras:
            if c.name == camera_name:
                cam_cfg = c
                break
        if cam_cfg is None:
            raise ValueError(f"Unknown camera {camera_name!r}; available: {[c.name for c in cfg.rig.cameras]}")

    cols, rows = cfg.chessboard.inner_corners
    target = ChessboardTarget(inner_rows=rows, inner_cols=cols, square_size_mm=cfg.chessboard.square_size_mm)
    corners_xyz = target.corners_xyz()

    if cfg.scene is not None:
        T_world_target = np.asarray(cfg.scene.T_world_target, dtype=np.float64)
    else:
        width_mm, height_mm = target.bounds()
        T_world_target = np.eye(4, dtype=np.float64)
        T_world_target[:3, 3] = np.array([-width_mm / 2.0, -height_mm / 2.0, 1000.0], dtype=np.float64)

    T_base_tcp = np.eye(4, dtype=np.float64)
    T_tcp_cam = np.asarray(cam_cfg.T_tcp_cam, dtype=np.float64)
    T_world_cam = T_base_tcp @ T_tcp_cam
    T_cam_target = invert_se3(T_world_cam) @ T_world_target

    cam = PinholeCamera(
        resolution=cam_cfg.image_size_px,
        K=np.asarray(cam_cfg.K, dtype=np.float64),
        dist=np.asarray(cam_cfg.dist, dtype=np.float64),
    )
    img = render_chessboard_image(cam, target, T_cam_target)
    corners_px, visible = project_corners_px(cam, corners_xyz, T_cam_target)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"frame={frame_index} cam={cam_cfg.name}")

    if bool(np.any(visible)):
        ax.scatter(corners_px[visible, 0], corners_px[visible, 1], s=12, c="lime", marker="o")
    if bool(np.any(~visible)):
        ax.scatter(corners_px[~visible, 0], corners_px[~visible, 1], s=12, c="red", marker="x")

    ax.set_xlim([-0.5, cam.resolution[0] - 0.5])
    ax.set_ylim([cam.resolution[1] - 0.5, -0.5])
    plt.show()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synthcal", description="Synthetic calibration dataset generator")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-config", help="Write an example config.yaml")
    p_init.add_argument("path", type=Path, help="Output path for config.yaml")

    p_gen = sub.add_parser("generate", help="Generate dataset (renders chessboard + GT corners)")
    p_gen.add_argument("config_yaml", type=Path, help="Input config.yaml")
    p_gen.add_argument("out_dir", type=Path, help="Output dataset directory")

    p_prev = sub.add_parser("preview", help="Render a single frame/camera and show a preview window")
    p_prev.add_argument("config_yaml", type=Path, help="Input config.yaml")
    p_prev.add_argument("--frame", type=int, default=0, help="Frame index (default: 0)")
    p_prev.add_argument(
        "--cam",
        type=str,
        default=None,
        help="Camera name (default: first camera in config)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init-config":
            return _cmd_init_config(args.path)
        if args.command == "generate":
            return _cmd_generate(args.config_yaml, args.out_dir)
        if args.command == "preview":
            return _cmd_preview(args.config_yaml, frame_index=args.frame, camera_name=args.cam)
        raise AssertionError(f"Unhandled command: {args.command}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
