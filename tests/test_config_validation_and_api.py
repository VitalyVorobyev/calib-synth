from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from synthcal.api import generate_dataset, render_frame_preview
from synthcal.io.config import ConfigError, load_config


def _minimal_config_dict(*, with_laser: bool) -> dict:
    cfg: dict = {
        "config_version": 1,
        "seed": 0,
        "units": {"length": "mm"},
        "dataset": {"name": "test_dataset", "num_frames": 1},
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
        cfg["laser"] = {
            "enabled": True,
            "plane_in_tcp": [1.0, 0.0, 0.0, 0.0],
            "stripe_width_px": 2,
            "stripe_intensity": 255,
        }
    return cfg


def test_load_config_missing_config_version_warns_and_assumes_v1(tmp_path: Path) -> None:
    cfg = _minimal_config_dict(with_laser=False)
    cfg.pop("config_version")
    cfg["version"] = 1

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = load_config(path)

    assert out.config_version == 1
    assert any("config_version missing" in str(w.message) for w in rec)


def test_load_config_rejects_unsupported_config_version(tmp_path: Path) -> None:
    cfg = _minimal_config_dict(with_laser=False)
    cfg["config_version"] = 2
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="Unsupported config_version"):
        load_config(path)


def test_load_config_validates_units_mm(tmp_path: Path) -> None:
    cfg = _minimal_config_dict(with_laser=False)
    cfg["units"]["length"] = "cm"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="units\\.length must be 'mm'"):
        load_config(path)


def test_load_config_validates_chessboard_min_size(tmp_path: Path) -> None:
    cfg = _minimal_config_dict(with_laser=False)
    cfg["chessboard"]["inner_corners"] = [1, 4]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="inner_corners"):
        load_config(path)


def test_load_config_validates_scenario_tilt_upper_bound(tmp_path: Path) -> None:
    cfg = _minimal_config_dict(with_laser=False)
    cfg["scenario"] = {"tilt_deg": {"min": 0.0, "max": 90.0}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="scenario\\.tilt_deg\\.max"):
        load_config(path)


def test_load_config_validates_laser_intensity_when_enabled(tmp_path: Path) -> None:
    cfg = _minimal_config_dict(with_laser=True)
    cfg["laser"]["stripe_intensity"] = 0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="laser\\.stripe_intensity"):
        load_config(path)


def test_example_configs_load() -> None:
    root = Path(__file__).resolve().parents[1]
    examples_dir = root / "examples"
    paths = sorted(examples_dir.glob("*.yaml"))
    assert paths, "No example configs found in examples/"
    for p in paths:
        cfg = load_config(p)
        assert cfg.config_version == 1


def test_api_generate_dataset_end_to_end_no_laser(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(_minimal_config_dict(with_laser=False), sort_keys=False), encoding="utf-8"
    )

    out_dir = tmp_path / "out"
    out = generate_dataset(cfg_path, out_dir)
    assert out == out_dir
    assert (out_dir / "manifest.yaml").is_file()
    assert (out_dir / "frames" / "frame_000000" / "cam00_target.png").is_file()
    assert not (out_dir / "frames" / "frame_000000" / "cam00_stripe.png").exists()


def test_api_generate_dataset_end_to_end_with_laser(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(_minimal_config_dict(with_laser=True), sort_keys=False), encoding="utf-8"
    )

    out_dir = tmp_path / "out"
    generate_dataset(cfg_path, out_dir)
    frame_dir = out_dir / "frames" / "frame_000000"
    assert (frame_dir / "cam00_stripe.png").is_file()
    assert (frame_dir / "cam00_stripe_centerline_px.npy").is_file()
    assert (frame_dir / "cam00_stripe_centerline_visible.npy").is_file()


def test_api_render_frame_preview_returns_arrays(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(_minimal_config_dict(with_laser=True), sort_keys=False), encoding="utf-8"
    )
    cfg = load_config(cfg_path)

    out = render_frame_preview(cfg, 0)
    assert out["frame_index"] == 0
    assert out["T_base_tcp"].shape == (4, 4)

    cams = out["cameras"]
    assert "cam00" in cams
    cam00 = cams["cam00"]

    assert cam00["target_image_u8"].dtype == np.uint8
    assert cam00["corners_px"].shape[1] == 2
    assert cam00["corners_visible"].dtype == bool

    assert cam00["stripe_image_u8"].dtype == np.uint8
    assert cam00["stripe_centerline_px"].shape[1] == 2
    assert cam00["stripe_centerline_visible"].dtype == bool
