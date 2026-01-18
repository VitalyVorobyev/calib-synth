from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

from synthcal.io.config import load_config
from synthcal.io.manifest import load_manifest


def _run_module(args: list[str]) -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), env.get("PYTHONPATH", "")]).strip(
        os.pathsep
    )
    subprocess.run([sys.executable, "-m", "synthcal", *args], check=True, env=env)


def test_init_config_writes_loadable_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _run_module(["init-config", str(config_path)])

    cfg = load_config(config_path)
    assert cfg.version == 1
    assert isinstance(cfg.seed, int)
    assert cfg.rig.cameras


def test_generate_creates_structure_and_manifest_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _run_module(["init-config", str(config_path)])
    cfg = load_config(config_path)

    out_dir = tmp_path / "out"
    _run_module(["generate", str(config_path), str(out_dir)])

    assert (out_dir / "config.yaml").is_file()
    assert (out_dir / "manifest.yaml").is_file()
    assert (out_dir / "rig" / "rig.yaml").is_file()
    assert (out_dir / "rig" / "cameras").is_dir()
    assert (out_dir / "frames").is_dir()

    manifest = load_manifest(out_dir / "manifest.yaml")
    assert manifest.seed == cfg.seed

    for rel in [
        manifest.paths.config_yaml,
        manifest.paths.manifest_yaml,
        manifest.paths.rig_yaml,
        manifest.paths.cameras_dir,
        manifest.paths.frames_dir,
    ]:
        assert (out_dir / rel).exists(), f"Missing path referenced by manifest: {rel}"

    for cam in manifest.cameras:
        assert (out_dir / cam.intrinsics_yaml).is_file()

    # Spot-check a couple of frame directories and expected outputs.
    assert (out_dir / "frames" / "frame_000000").is_dir()
    assert (out_dir / "frames" / f"frame_{cfg.dataset.num_frames - 1:06d}").is_dir()
    assert (out_dir / "frames" / "frame_000000" / "T_base_tcp.npy").is_file()
    for cam in cfg.rig.cameras:
        assert (out_dir / "frames" / "frame_000000" / f"{cam.name}_target.png").is_file()
        assert (out_dir / "frames" / "frame_000000" / f"{cam.name}_corners_px.npy").is_file()
        assert (out_dir / "frames" / "frame_000000" / f"{cam.name}_corners_visible.npy").is_file()


def test_generate_omits_laser_outputs_when_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _run_module(["init-config", str(config_path)])

    # Disable laser by setting the section to null (also validates that config loader accepts it).
    cfg_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg_data["laser"] = None
    config_path.write_text(
        yaml.safe_dump(cfg_data, sort_keys=False, indent=2, default_flow_style=False),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    _run_module(["generate", str(config_path), str(out_dir)])

    manifest_path = out_dir / "manifest.yaml"
    manifest_raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert "laser" not in manifest_raw
    assert "stripe_image" not in manifest_raw["layout"]
    assert "stripe_centerline_px_npy" not in manifest_raw["layout"]
    assert "stripe_centerline_visible_npy" not in manifest_raw["layout"]

    manifest = load_manifest(manifest_path)
    assert manifest.laser is None
    assert manifest.layout.stripe_image is None
    assert manifest.layout.stripe_centerline_px_npy is None
    assert manifest.layout.stripe_centerline_visible_npy is None


def test_generate_includes_laser_outputs_when_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _run_module(["init-config", str(config_path)])

    cfg_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg_data["laser"] = {
        "enabled": True,
        "plane_in_tcp": [1.0, 0.0, 0.0, 0.0],
        "stripe_width_px": 3,
        "stripe_intensity": 255,
    }
    config_path.write_text(
        yaml.safe_dump(cfg_data, sort_keys=False, indent=2, default_flow_style=False),
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    assert cfg.laser is not None and cfg.laser.enabled

    out_dir = tmp_path / "out"
    _run_module(["generate", str(config_path), str(out_dir)])

    manifest_path = out_dir / "manifest.yaml"
    manifest_raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert "laser" in manifest_raw
    assert "stripe_image" in manifest_raw["layout"]
    assert "stripe_centerline_px_npy" in manifest_raw["layout"]
    assert "stripe_centerline_visible_npy" in manifest_raw["layout"]

    frame_dir = out_dir / "frames" / "frame_000000"
    for cam in cfg.rig.cameras:
        assert (frame_dir / f"{cam.name}_stripe.png").is_file()
        assert (frame_dir / f"{cam.name}_stripe_centerline_px.npy").is_file()
        assert (frame_dir / f"{cam.name}_stripe_centerline_visible.npy").is_file()
