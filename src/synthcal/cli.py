"""Command-line interface.

Entry point: `python -m synthcal ...`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from synthcal import __version__
from synthcal.io.config import SynthCalConfig, load_config, save_config
from synthcal.io.manifest import (
    ManifestCamera,
    ManifestGenerator,
    ManifestLayout,
    ManifestLaser,
    ManifestPaths,
    SynthCalManifest,
    save_manifest,
    utc_now_iso8601,
)


def _cmd_init_config(path: Path) -> int:
    cfg = SynthCalConfig.example()
    save_config(cfg, path)
    return 0


def _write_rig_files(cfg: SynthCalConfig, out_dir: Path) -> None:
    rig_dir = out_dir / "rig"
    cams_dir = rig_dir / "cameras"
    cams_dir.mkdir(parents=True, exist_ok=True)

    rig_yaml = {
        "version": 1,
        "units": {"length": "mm"},
        "cameras": [
            {"name": cam.name, "intrinsics_yaml": f"cameras/{cam.name}.yaml"}
            for cam in cfg.rig.cameras
        ],
        "notes": "Placeholder rig file (v0). Extrinsics will be added later.",
    }
    (rig_dir / "rig.yaml").write_text(
        _yaml_dump(rig_yaml),
        encoding="utf-8",
    )

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


def _write_frame_dirs(cfg: SynthCalConfig, out_dir: Path) -> None:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for frame_index in range(cfg.dataset.num_frames):
        frame_dir = frames_dir / f"frame_{frame_index:06d}"
        for cam in cfg.rig.cameras:
            (frame_dir / f"cam_{cam.name}").mkdir(parents=True, exist_ok=True)


def _cmd_generate(config_path: Path, out_dir: Path) -> int:
    cfg = load_config(config_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    save_config(cfg, out_dir / "config.yaml")
    _write_rig_files(cfg, out_dir)
    _write_frame_dirs(cfg, out_dir)

    laser_enabled = cfg.laser is not None and cfg.laser.enabled
    laser_manifest = None
    if laser_enabled:
        laser = cfg.laser
        assert laser is not None
        laser_manifest = ManifestLaser(
            enabled=True,
            plane_in_tcp=laser.plane_in_tcp,
            stripe_width_px=laser.stripe_width_px,
            stripe_intensity=laser.stripe_intensity,
        )

    manifest = SynthCalManifest(
        manifest_version=1,
        created_utc=utc_now_iso8601(),
        generator=ManifestGenerator(name="synthcal", version=__version__),
        seed=cfg.seed,
        units={"length": "mm"},
        dataset={"name": cfg.dataset.name, "num_frames": cfg.dataset.num_frames},
        laser=laser_manifest,
        paths=ManifestPaths(
            config_yaml="config.yaml",
            manifest_yaml="manifest.yaml",
            rig_yaml="rig/rig.yaml",
            cameras_dir="rig/cameras",
            frames_dir="frames",
        ),
        cameras=tuple(
            ManifestCamera(
                name=cam.name,
                intrinsics_yaml=f"rig/cameras/{cam.name}.yaml",
                image_size_px=cam.image_size_px,
            )
            for cam in cfg.rig.cameras
        ),
        layout=ManifestLayout.v1_default(include_laser=laser_enabled),
    )
    save_manifest(manifest, out_dir / "manifest.yaml")
    return 0


def _yaml_dump(data: object) -> str:
    # Local helper to avoid a PyYAML dependency from "util".
    import yaml

    return yaml.safe_dump(
        data,
        sort_keys=False,
        indent=2,
        default_flow_style=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synthcal", description="Synthetic calibration dataset generator")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-config", help="Write an example config.yaml")
    p_init.add_argument("path", type=Path, help="Output path for config.yaml")

    p_gen = sub.add_parser("generate", help="Create dataset folder structure + manifest (no rendering yet)")
    p_gen.add_argument("config_yaml", type=Path, help="Input config.yaml")
    p_gen.add_argument("out_dir", type=Path, help="Output dataset directory")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init-config":
            return _cmd_init_config(args.path)
        if args.command == "generate":
            return _cmd_generate(args.config_yaml, args.out_dir)
        raise AssertionError(f"Unhandled command: {args.command}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
