from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

import synthcal
from synthcal.api import generate_dataset, render_frame_preview
from synthcal.io.config import SynthCalConfig, load_config
from synthcal.io.manifest import load_manifest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_yaml(data: dict, path: Path) -> None:
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, indent=2, default_flow_style=False), encoding="utf-8"
    )


def _assert_file_nonempty(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing file: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Empty file: {path}")


def _check_png(
    path: Path,
    *,
    expected_size_px: tuple[int, int] | None,
    require_nonzero: bool,
) -> None:
    _assert_file_nonempty(path)
    with Image.open(path) as im:
        im.load()
        if im.mode != "L":
            raise RuntimeError(f"Expected grayscale PNG (mode 'L'), got {im.mode} in {path}")
        if expected_size_px is not None and im.size != expected_size_px:
            msg = f"Unexpected PNG size for {path}: got {im.size}, expected {expected_size_px}"
            raise RuntimeError(msg)
        arr = np.asarray(im)
    if require_nonzero and int(arr.max()) <= 0:
        raise RuntimeError(f"Expected non-zero pixels in {path}")


def _check_generated_dataset(out_dir: Path, *, with_laser: bool) -> None:
    manifest = load_manifest(out_dir / "manifest.yaml")
    if manifest.manifest_version != 1:
        raise RuntimeError(f"Unexpected manifest_version: {manifest.manifest_version}")

    frame_dir = out_dir / "frames" / "frame_000000"
    _assert_file_nonempty(frame_dir / "T_base_tcp.npy")
    T_base_tcp = np.load(frame_dir / "T_base_tcp.npy")
    if T_base_tcp.shape != (4, 4):
        raise RuntimeError(f"Unexpected T_base_tcp shape: {T_base_tcp.shape}")

    target_png = frame_dir / "cam00_target.png"
    _check_png(target_png, expected_size_px=(160, 120), require_nonzero=True)
    _assert_file_nonempty(frame_dir / "cam00_corners_px.npy")
    _assert_file_nonempty(frame_dir / "cam00_corners_visible.npy")
    corners_px = np.load(frame_dir / "cam00_corners_px.npy")
    corners_vis = np.load(frame_dir / "cam00_corners_visible.npy")
    if corners_px.ndim != 2 or corners_px.shape[1] != 2:
        raise RuntimeError(f"Unexpected corners_px shape: {corners_px.shape}")
    if corners_vis.ndim != 1 or corners_vis.shape[0] != corners_px.shape[0]:
        raise RuntimeError(f"Unexpected corners_visible shape: {corners_vis.shape}")

    stripe_png = frame_dir / "cam00_stripe.png"
    if with_laser:
        _check_png(stripe_png, expected_size_px=(160, 120), require_nonzero=True)
        _assert_file_nonempty(frame_dir / "cam00_stripe_centerline_px.npy")
        _assert_file_nonempty(frame_dir / "cam00_stripe_centerline_visible.npy")
        stripe_px = np.load(frame_dir / "cam00_stripe_centerline_px.npy")
        stripe_vis = np.load(frame_dir / "cam00_stripe_centerline_visible.npy")
        if stripe_px.ndim != 2 or stripe_px.shape[1] != 2:
            raise RuntimeError(f"Unexpected stripe_centerline_px shape: {stripe_px.shape}")
        if stripe_vis.ndim != 1 or stripe_vis.shape[0] != stripe_px.shape[0]:
            raise RuntimeError(f"Unexpected stripe_centerline_visible shape: {stripe_vis.shape}")
    else:
        if stripe_png.exists():
            raise RuntimeError(f"Unexpected stripe output when laser is disabled: {stripe_png}")


def _check_examples_load(root: Path) -> None:
    examples_dir = root / "examples"
    paths = sorted(examples_dir.glob("*.yaml"))
    if not paths:
        raise RuntimeError("No example configs found in examples/")
    for p in paths:
        _ = load_config(p)


def _make_minimal_cfg(*, with_laser: bool) -> SynthCalConfig:
    cfg_dict: dict = {
        "config_version": 1,
        "seed": 0,
        "units": {"length": "mm"},
        "dataset": {"name": "release_check", "num_frames": 1},
        "rig": {
            "cameras": [
                {
                    "name": "cam00",
                    "image_size_px": [160, 120],
                    "K": [[200.0, 0.0, 80.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]],
                    "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "T_tcp_cam": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                }
            ]
        },
        "chessboard": {"inner_corners": [5, 4], "square_size_mm": 25.0},
        "effects": {"enabled": True, "blur_sigma_px": 0.0, "noise_sigma": 0.0},
    }
    if with_laser:
        cfg_dict["laser"] = {
            "enabled": True,
            "plane_in_tcp": [1.0, 0.0, 0.0, 0.0],
            "stripe_width_px": 2,
            "stripe_intensity": 255,
        }
    cfg = SynthCalConfig.from_dict(cfg_dict)
    if with_laser and cfg.laser is None:
        raise RuntimeError("Expected laser to be enabled in config")
    return cfg


def _check_preview(cfg: SynthCalConfig, *, expect_stripe: bool) -> None:
    out = render_frame_preview(cfg, 0)
    cams = out["cameras"]
    cam00 = cams["cam00"]
    target_shape = cam00["target_image_u8"].shape
    if target_shape != (120, 160):
        raise RuntimeError(f"Unexpected preview target image shape: {target_shape}")
    if expect_stripe:
        if "stripe_image_u8" not in cam00:
            raise RuntimeError("Expected stripe_image_u8 in preview output")
        stripe_shape = cam00["stripe_image_u8"].shape
        if stripe_shape != (120, 160):
            raise RuntimeError(f"Unexpected preview stripe image shape: {stripe_shape}")
    else:
        if "stripe_image_u8" in cam00:
            raise RuntimeError("Unexpected stripe output in preview when laser is disabled")


def main() -> int:
    root = _repo_root()

    print(f"synthcal.__version__ = {synthcal.__version__}")

    print("Checking example configs...")
    _check_examples_load(root)

    cfg_no_laser = _make_minimal_cfg(with_laser=False)
    cfg_with_laser = _make_minimal_cfg(with_laser=True)

    print("Checking API preview...")
    _check_preview(cfg_no_laser, expect_stripe=False)
    _check_preview(cfg_with_laser, expect_stripe=True)

    with tempfile.TemporaryDirectory(prefix="synthcal_release_check_") as td:
        td_path = Path(td)

        print("Generating tiny dataset (laser off)...")
        cfg_path = td_path / "config_no_laser.yaml"
        _write_yaml(cfg_no_laser.to_dict(), cfg_path)
        out_dir = td_path / "out_no_laser"
        generate_dataset(cfg_path, out_dir)
        _check_generated_dataset(out_dir, with_laser=False)

        print("Generating tiny dataset (laser on)...")
        cfg_path = td_path / "config_with_laser.yaml"
        _write_yaml(cfg_with_laser.to_dict(), cfg_path)
        out_dir = td_path / "out_with_laser"
        generate_dataset(cfg_path, out_dir)
        _check_generated_dataset(out_dir, with_laser=True)

    # Ensure the source tree doesn't accidentally diverge from the on-disk normalized config format.
    _ = replace(cfg_with_laser, seed=123).to_dict()

    print("OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"release_check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
