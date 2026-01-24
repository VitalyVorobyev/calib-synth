"""Config schema (v1) and YAML helpers.

The top-level `config_version` is versioned and validated on load. Future releases may provide
upgrade helpers for older versions.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from synthcal.effects.config import EffectsConfig
from synthcal.scenario.config import InViewConfig, RangeConfig, ScenarioConfig
from synthcal.util import require_ascii_filename_component


class ConfigError(ValueError):
    """Raised when a config file is invalid or cannot be parsed."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if value is None:
        raise ConfigError(f"{label} is required")
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping/object")
    return value


def _require_int(value: Any, *, label: str) -> int:
    if value is None:
        raise ConfigError(f"{label} is required")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    return value


def _require_number(value: Any, *, label: str) -> float:
    if value is None:
        raise ConfigError(f"{label} is required")
    if not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number")
    return float(value)


def _require_seq(value: Any, *, label: str) -> Sequence[Any]:
    if value is None:
        raise ConfigError(f"{label} is required")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ConfigError(f"{label} must be a sequence/list")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if value is None:
        raise ConfigError(f"{label} is required")
    if not isinstance(value, bool):
        raise ConfigError(f"{label} must be a bool")
    return value


def _require_mat44(value: Any, *, label: str) -> tuple[tuple[float, float, float, float], ...]:
    rows = _require_seq(value, label=label)
    if len(rows) != 4:
        raise ConfigError(f"{label} must be a 4x4 matrix")
    out: list[tuple[float, float, float, float]] = []
    for r in range(4):
        row = _require_seq(rows[r], label=f"{label}[{r}]")
        if len(row) != 4:
            raise ConfigError(f"{label} must be a 4x4 matrix")
        out.append(
            (
                _require_number(row[0], label=f"{label}[{r}][0]"),
                _require_number(row[1], label=f"{label}[{r}][1]"),
                _require_number(row[2], label=f"{label}[{r}][2]"),
                _require_number(row[3], label=f"{label}[{r}][3]"),
            )
        )
    return tuple(out)


def _mat44_identity() -> tuple[tuple[float, float, float, float], ...]:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset-level parameters."""

    name: str
    num_frames: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetConfig:
        data = _require_mapping(data, label="dataset")
        name = data.get("name", "dataset")
        if not isinstance(name, str) or not name:
            raise ConfigError("dataset.name must be a non-empty string")
        num_frames = _require_int(data.get("num_frames"), label="dataset.num_frames")
        if num_frames <= 0:
            raise ConfigError("dataset.num_frames must be > 0")
        return cls(name=name, num_frames=num_frames)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "num_frames": self.num_frames}


@dataclass(frozen=True)
class ChessboardConfig:
    """Chessboard calibration target description.

    `inner_corners` uses OpenCV convention: (cols, rows).
    """

    inner_corners: tuple[int, int]
    square_size_mm: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChessboardConfig:
        data = _require_mapping(data, label="chessboard")
        inner = _require_seq(data.get("inner_corners"), label="chessboard.inner_corners")
        if len(inner) != 2:
            raise ConfigError("chessboard.inner_corners must have length 2: [cols, rows]")
        cols = _require_int(inner[0], label="chessboard.inner_corners[0]")
        rows = _require_int(inner[1], label="chessboard.inner_corners[1]")
        if cols < 2 or rows < 2:
            raise ConfigError("chessboard.inner_corners values must be >= 2")
        square_size_mm = _require_number(
            data.get("square_size_mm"), label="chessboard.square_size_mm"
        )
        if square_size_mm <= 0.0:
            raise ConfigError("chessboard.square_size_mm must be > 0")
        return cls(inner_corners=(cols, rows), square_size_mm=square_size_mm)

    def to_dict(self) -> dict[str, Any]:
        cols, rows = self.inner_corners
        return {"inner_corners": [cols, rows], "square_size_mm": self.square_size_mm}


@dataclass(frozen=True)
class CameraConfig:
    """Per-camera configuration.

    Intrinsics follow OpenCV conventions: `K` (3x3) and `dist` = (k1,k2,k3,p1,p2).
    """

    name: str
    image_size_px: tuple[int, int]
    K: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    dist: tuple[float, float, float, float, float]
    T_tcp_cam: tuple[tuple[float, float, float, float], ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CameraConfig:
        data = _require_mapping(data, label="camera")
        name = require_ascii_filename_component(data.get("name"), label="camera.name")

        size = _require_seq(data.get("image_size_px"), label=f"camera[{name}].image_size_px")
        if len(size) != 2:
            raise ConfigError(f"camera[{name}].image_size_px must be [width, height]")
        width = _require_int(size[0], label=f"camera[{name}].image_size_px[0]")
        height = _require_int(size[1], label=f"camera[{name}].image_size_px[1]")
        if width <= 0 or height <= 0:
            raise ConfigError(f"camera[{name}].image_size_px values must be > 0")

        K_raw = _require_seq(data.get("K"), label=f"camera[{name}].K")
        if len(K_raw) != 3:
            raise ConfigError(f"camera[{name}].K must be 3x3")
        K_rows: list[tuple[float, float, float]] = []
        for r in range(3):
            row = _require_seq(K_raw[r], label=f"camera[{name}].K[{r}]")
            if len(row) != 3:
                raise ConfigError(f"camera[{name}].K must be 3x3")
            K_rows.append(
                (
                    _require_number(row[0], label=f"camera[{name}].K[{r}][0]"),
                    _require_number(row[1], label=f"camera[{name}].K[{r}][1]"),
                    _require_number(row[2], label=f"camera[{name}].K[{r}][2]"),
                )
            )

        dist_raw = _require_seq(data.get("dist"), label=f"camera[{name}].dist")
        if len(dist_raw) != 5:
            raise ConfigError(f"camera[{name}].dist must have 5 values: [k1,k2,k3,p1,p2]")
        dist = tuple(
            _require_number(dist_raw[i], label=f"camera[{name}].dist[{i}]") for i in range(5)
        )

        T_tcp_cam_raw = data.get("T_tcp_cam", None)
        T_tcp_cam = (
            _mat44_identity()
            if T_tcp_cam_raw is None
            else _require_mat44(T_tcp_cam_raw, label=f"camera[{name}].T_tcp_cam")
        )
        return cls(
            name=name,
            image_size_px=(width, height),
            K=(K_rows[0], K_rows[1], K_rows[2]),
            dist=dist,  # type: ignore[arg-type]
            T_tcp_cam=T_tcp_cam,
        )

    def to_dict(self) -> dict[str, Any]:
        width, height = self.image_size_px
        return {
            "name": self.name,
            "image_size_px": [width, height],
            "K": [list(row) for row in self.K],
            "dist": list(self.dist),
            "T_tcp_cam": [list(row) for row in self.T_tcp_cam],
        }


@dataclass(frozen=True)
class RigConfig:
    """Rig configuration for an eye-in-hand multi-camera setup."""

    cameras: tuple[CameraConfig, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RigConfig:
        data = _require_mapping(data, label="rig")
        cams_raw = _require_seq(data.get("cameras"), label="rig.cameras")
        cameras = tuple(CameraConfig.from_dict(c) for c in cams_raw)
        if not cameras:
            raise ConfigError("rig.cameras must contain at least one camera")
        names = [c.name for c in cameras]
        if len(set(names)) != len(names):
            raise ConfigError("rig.cameras camera names must be unique")
        return cls(cameras=cameras)

    def to_dict(self) -> dict[str, Any]:
        return {"cameras": [c.to_dict() for c in self.cameras]}


@dataclass(frozen=True)
class SceneConfig:
    """Static scene description."""

    T_world_target: tuple[tuple[float, float, float, float], ...]

    @classmethod
    def from_optional(cls, data: Any) -> SceneConfig | None:
        if data is None:
            return None
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SceneConfig:
        data = _require_mapping(data, label="scene")
        T_world_target = _require_mat44(data.get("T_world_target"), label="scene.T_world_target")
        return cls(T_world_target=T_world_target)

    def to_dict(self) -> dict[str, Any]:
        return {"T_world_target": [list(row) for row in self.T_world_target]}


@dataclass(frozen=True)
class LaserConfig:
    """Laser configuration (optional).

    When `enabled` is false, laser outputs (stripe image / centerline) are not produced.
    """

    enabled: bool
    plane_in_tcp: tuple[float, float, float, float]
    stripe_width_px: int
    stripe_intensity: int

    @classmethod
    def from_optional(cls, data: Any) -> LaserConfig | None:
        if data is None:
            return None
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LaserConfig:
        data = _require_mapping(data, label="laser")
        enabled = _require_bool(data.get("enabled", False), label="laser.enabled")

        plane_raw = data.get("plane_in_tcp", [0.0, 0.0, 1.0, 0.0])
        plane_seq = _require_seq(plane_raw, label="laser.plane_in_tcp")
        if len(plane_seq) != 4:
            raise ConfigError("laser.plane_in_tcp must have length 4: [a, b, c, d]")
        plane = tuple(
            _require_number(plane_seq[i], label=f"laser.plane_in_tcp[{i}]") for i in range(4)
        )

        stripe_width_px = _require_int(
            data.get("stripe_width_px", 3), label="laser.stripe_width_px"
        )
        if stripe_width_px <= 0:
            raise ConfigError("laser.stripe_width_px must be > 0")

        stripe_intensity = _require_int(
            data.get("stripe_intensity", 255), label="laser.stripe_intensity"
        )
        if not (0 <= stripe_intensity <= 255):
            raise ConfigError("laser.stripe_intensity must be in [0, 255]")
        if enabled and stripe_intensity < 1:
            raise ConfigError("laser.stripe_intensity must be in [1, 255] when laser is enabled")

        if enabled and all(abs(v) < 1e-15 for v in plane[:3]):
            raise ConfigError("laser.plane_in_tcp normal must be non-zero when laser is enabled")

        return cls(
            enabled=enabled,
            plane_in_tcp=plane,  # type: ignore[arg-type]
            stripe_width_px=stripe_width_px,
            stripe_intensity=stripe_intensity,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "plane_in_tcp": list(self.plane_in_tcp),
            "stripe_width_px": self.stripe_width_px,
            "stripe_intensity": self.stripe_intensity,
        }


def _effects_from_optional(data: Any) -> EffectsConfig:
    if data is None:
        return EffectsConfig()
    data = _require_mapping(data, label="effects")

    enabled = _require_bool(data.get("enabled", True), label="effects.enabled")
    blur_sigma_px = _require_number(data.get("blur_sigma_px", 0.0), label="effects.blur_sigma_px")
    noise_sigma = _require_number(data.get("noise_sigma", 0.0), label="effects.noise_sigma")
    clamp_min = _require_number(data.get("clamp_min", 0.0), label="effects.clamp_min")
    clamp_max = _require_number(data.get("clamp_max", 255.0), label="effects.clamp_max")

    if blur_sigma_px < 0.0:
        raise ConfigError("effects.blur_sigma_px must be >= 0")
    if noise_sigma < 0.0:
        raise ConfigError("effects.noise_sigma must be >= 0")
    if clamp_min < 0.0 or clamp_max > 255.0 or clamp_min > clamp_max:
        raise ConfigError("effects clamp range must satisfy 0 <= clamp_min <= clamp_max <= 255")

    return EffectsConfig(
        enabled=enabled,
        blur_sigma_px=blur_sigma_px,
        noise_sigma=noise_sigma,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )


def _range_from_optional(
    data: Any, *, label: str, default_min: float, default_max: float
) -> RangeConfig:
    if data is None:
        return RangeConfig(min=default_min, max=default_max)
    data = _require_mapping(data, label=label)
    min_v = _require_number(data.get("min", default_min), label=f"{label}.min")
    max_v = _require_number(data.get("max", default_max), label=f"{label}.max")
    if max_v < min_v:
        raise ConfigError(f"{label}.max must be >= {label}.min")
    return RangeConfig(min=min_v, max=max_v)


def _pose_sampler_preset_defaults(name: str, *, label: str) -> dict[str, Any]:
    preset = name.strip().lower()
    if preset == "easy":
        return {
            "distance_mm": {"min": 800.0, "max": 1200.0},
            "tilt_deg": {"min": 0.0, "max": 20.0},
            "xy_offset_frac": {"min": -0.05, "max": 0.05},
            "in_view": {"margin_px": 20, "require_all_cameras": True, "min_cameras_visible": 1},
            "max_attempts_per_frame": 200,
        }
    if preset == "medium":
        return {
            "distance_mm": {"min": 600.0, "max": 1200.0},
            "tilt_deg": {"min": 0.0, "max": 45.0},
            "xy_offset_frac": {"min": -0.1, "max": 0.1},
            "in_view": {"margin_px": 20, "require_all_cameras": True, "min_cameras_visible": 1},
            "max_attempts_per_frame": 500,
        }
    if preset == "hard":
        return {
            "distance_mm": {"min": 400.0, "max": 1000.0},
            "tilt_deg": {"min": 15.0, "max": 60.0},
            "xy_offset_frac": {"min": -0.2, "max": 0.2},
            "in_view": {"margin_px": 10, "require_all_cameras": False, "min_cameras_visible": 1},
            "max_attempts_per_frame": 800,
        }
    raise ConfigError(f"Unknown {label}.preset {name!r}; expected one of: easy, medium, hard")


def _pose_sampler_from_optional(data: Any, *, label: str) -> ScenarioConfig | None:
    if data is None:
        return None
    data = _require_mapping(data, label=label)

    preset = data.get("preset", None)
    preset_defaults: dict[str, Any] = {}
    if isinstance(preset, str) and preset.strip():
        preset_defaults = _pose_sampler_preset_defaults(preset, label=label)

    num_frames_raw = data.get("num_frames", preset_defaults.get("num_frames", None))
    num_frames = None
    if num_frames_raw is not None:
        num_frames = _require_int(num_frames_raw, label=f"{label}.num_frames")
        if num_frames <= 0:
            raise ConfigError(f"{label}.num_frames must be > 0")

    default_distance = preset_defaults.get("distance_mm", {"min": 400.0, "max": 1200.0})
    default_tilt = preset_defaults.get("tilt_deg", {"min": 0.0, "max": 45.0})
    default_yaw = preset_defaults.get("yaw_deg", {"min": -180.0, "max": 180.0})
    default_roll = preset_defaults.get("roll_deg", {"min": -180.0, "max": 180.0})
    default_xy = preset_defaults.get("xy_offset_frac", {"min": 0.0, "max": 0.0})

    distance_mm = _range_from_optional(
        data.get("distance_mm"),
        label=f"{label}.distance_mm",
        default_min=float(default_distance.get("min", 400.0)),
        default_max=float(default_distance.get("max", 1200.0)),
    )
    if distance_mm.min <= 0.0:
        raise ConfigError(f"{label}.distance_mm.min must be > 0")
    tilt_deg = _range_from_optional(
        data.get("tilt_deg"),
        label=f"{label}.tilt_deg",
        default_min=float(default_tilt.get("min", 0.0)),
        default_max=float(default_tilt.get("max", 45.0)),
    )
    if tilt_deg.min < 0.0:
        raise ConfigError(f"{label}.tilt_deg.min must be >= 0")
    if tilt_deg.max > 89.0:
        raise ConfigError(f"{label}.tilt_deg.max must be <= 89")
    yaw_deg = _range_from_optional(
        data.get("yaw_deg"),
        label=f"{label}.yaw_deg",
        default_min=float(default_yaw.get("min", -180.0)),
        default_max=float(default_yaw.get("max", 180.0)),
    )
    roll_deg = _range_from_optional(
        data.get("roll_deg"),
        label=f"{label}.roll_deg",
        default_min=float(default_roll.get("min", -180.0)),
        default_max=float(default_roll.get("max", 180.0)),
    )
    xy_offset_frac = _range_from_optional(
        data.get("xy_offset_frac"),
        label=f"{label}.xy_offset_frac",
        default_min=float(default_xy.get("min", 0.0)),
        default_max=float(default_xy.get("max", 0.0)),
    )

    max_attempts = _require_int(
        data.get("max_attempts_per_frame", preset_defaults.get("max_attempts_per_frame", 500)),
        label=f"{label}.max_attempts_per_frame",
    )
    if max_attempts <= 0:
        raise ConfigError(f"{label}.max_attempts_per_frame must be > 0")

    default_in_view = preset_defaults.get("in_view", {})
    in_view_raw = data.get("in_view", {})
    in_view_map = _require_mapping(in_view_raw, label=f"{label}.in_view")
    margin_px = _require_int(
        in_view_map.get("margin_px", default_in_view.get("margin_px", 20)),
        label=f"{label}.in_view.margin_px",
    )
    if margin_px < 0:
        raise ConfigError(f"{label}.in_view.margin_px must be >= 0")
    require_all_cameras = _require_bool(
        in_view_map.get("require_all_cameras", default_in_view.get("require_all_cameras", True)),
        label=f"{label}.in_view.require_all_cameras",
    )
    min_cameras_visible = _require_int(
        in_view_map.get("min_cameras_visible", default_in_view.get("min_cameras_visible", 1)),
        label=f"{label}.in_view.min_cameras_visible",
    )
    if min_cameras_visible <= 0:
        raise ConfigError(f"{label}.in_view.min_cameras_visible must be > 0")

    in_view = InViewConfig(
        margin_px=margin_px,
        require_all_cameras=require_all_cameras,
        min_cameras_visible=min_cameras_visible,
    )

    preset_out = preset.strip() if isinstance(preset, str) and preset.strip() else None
    return ScenarioConfig(
        num_frames=num_frames,
        distance_mm=distance_mm,
        tilt_deg=tilt_deg,
        yaw_deg=yaw_deg,
        roll_deg=roll_deg,
        xy_offset_frac=xy_offset_frac,
        in_view=in_view,
        max_attempts_per_frame=max_attempts,
        preset=preset_out,
    )


def _scenario_from_optional(data: Any) -> ScenarioConfig | None:
    return _pose_sampler_from_optional(data, label="scenario")


@dataclass(frozen=True)
class UnitsConfig:
    """Units for geometric quantities in the config.

    synthcal currently supports millimeters (mm) only.
    """

    length: str = "mm"

    @classmethod
    def from_optional(cls, data: Any) -> UnitsConfig:
        if data is None:
            return cls(length="mm")
        data = _require_mapping(data, label="units")
        length = data.get("length", "mm")
        if not isinstance(length, str) or not length:
            raise ConfigError("units.length must be a non-empty string")
        if length != "mm":
            raise ConfigError("units.length must be 'mm'")
        return cls(length=length)

    def to_dict(self) -> dict[str, Any]:
        return {"length": self.length}


@dataclass(frozen=True)
class SynthCalConfig:
    """Top-level config file."""

    config_version: int
    seed: int
    units: UnitsConfig
    dataset: DatasetConfig
    rig: RigConfig
    chessboard: ChessboardConfig
    effects: EffectsConfig
    laser: LaserConfig | None
    scene: SceneConfig | None
    scenario: ScenarioConfig | None
    target_sampling: ScenarioConfig | None

    @classmethod
    def example(cls) -> SynthCalConfig:
        """Return a small example config with sane defaults."""

        # A single camera with reasonable-looking intrinsics for 1280x720.
        cam0 = CameraConfig(
            name="cam00",
            image_size_px=(640, 480),
            K=((800.0, 0.0, 320.0), (0.0, 800.0, 240.0), (0.0, 0.0, 1.0)),
            dist=(0.0, 0.0, 0.0, 0.0, 0.0),
            T_tcp_cam=_mat44_identity(),
        )
        cam1 = CameraConfig(
            name="cam01",
            image_size_px=(640, 480),
            K=((800.0, 0.0, 320.0), (0.0, 800.0, 240.0), (0.0, 0.0, 1.0)),
            dist=(0.0, 0.0, 0.0, 0.0, 0.0),
            T_tcp_cam=(
                (1.0, 0.0, 0.0, 100.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )

        chessboard = ChessboardConfig(inner_corners=(9, 6), square_size_mm=25.0)

        return cls(
            config_version=1,
            seed=0,
            units=UnitsConfig(),
            dataset=DatasetConfig(name="example_dataset", num_frames=1),
            rig=RigConfig(cameras=(cam0, cam1)),
            chessboard=chessboard,
            effects=EffectsConfig(),
            laser=None,
            scene=None,
            scenario=None,
            target_sampling=ScenarioConfig(
                num_frames=None,
                distance_mm=RangeConfig(min=800.0, max=1200.0),
                tilt_deg=RangeConfig(min=0.0, max=20.0),
                yaw_deg=RangeConfig(min=-180.0, max=180.0),
                roll_deg=RangeConfig(min=-180.0, max=180.0),
                xy_offset_frac=RangeConfig(min=-0.05, max=0.05),
                in_view=InViewConfig(margin_px=20, require_all_cameras=True, min_cameras_visible=1),
                max_attempts_per_frame=200,
                preset="easy",
            ),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SynthCalConfig:
        data = _require_mapping(data, label="config")

        config_version_raw = data.get("config_version", None)
        if config_version_raw is None:
            legacy = data.get("version", None)
            if legacy is not None:
                warnings.warn(
                    "config_version missing; using legacy 'version' field (assumed v1)",
                    UserWarning,
                    stacklevel=2,
                )
                config_version_raw = legacy
            else:
                warnings.warn(
                    "config_version missing; assuming v1",
                    UserWarning,
                    stacklevel=2,
                )
                config_version_raw = 1

        config_version = _require_int(config_version_raw, label="config_version")
        if config_version != 1:
            raise ConfigError(
                f"Unsupported config_version {config_version}; supported versions: [1]"
            )

        seed = _require_int(data.get("seed"), label="seed")
        if seed < 0:
            raise ConfigError("seed must be >= 0")
        units = UnitsConfig.from_optional(data.get("units"))
        dataset = DatasetConfig.from_dict(data.get("dataset"))
        rig = RigConfig.from_dict(data.get("rig"))
        chessboard = ChessboardConfig.from_dict(data.get("chessboard"))
        effects = _effects_from_optional(data.get("effects"))
        scenario = _scenario_from_optional(data.get("scenario"))
        target_sampling = _pose_sampler_from_optional(
            data.get("target_sampling"), label="target_sampling"
        )
        laser = LaserConfig.from_optional(data.get("laser"))
        if laser is not None and laser.enabled is False:
            laser = None
        scene = SceneConfig.from_optional(data.get("scene"))
        if scenario is not None and target_sampling is not None:
            raise ConfigError("scenario and target_sampling are mutually exclusive; pick one")
        if target_sampling is not None and scene is not None:
            raise ConfigError("scene and target_sampling are mutually exclusive; pick one")
        if scenario is not None and scenario.num_frames is not None:
            dataset = DatasetConfig(name=dataset.name, num_frames=scenario.num_frames)
        if target_sampling is not None and target_sampling.num_frames is not None:
            dataset = DatasetConfig(name=dataset.name, num_frames=target_sampling.num_frames)
        return cls(
            config_version=config_version,
            seed=seed,
            units=units,
            dataset=dataset,
            rig=rig,
            chessboard=chessboard,
            effects=effects,
            laser=laser,
            scene=scene,
            scenario=scenario,
            target_sampling=target_sampling,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "config_version": self.config_version,
            "seed": self.seed,
            "units": self.units.to_dict(),
            "dataset": self.dataset.to_dict(),
            "rig": self.rig.to_dict(),
            "chessboard": self.chessboard.to_dict(),
            "effects": self.effects.to_dict(),
        }
        if self.scenario is not None:
            data["scenario"] = self.scenario.to_dict()
        if self.target_sampling is not None:
            data["target_sampling"] = self.target_sampling.to_dict()
        if self.laser is not None and self.laser.enabled:
            data["laser"] = self.laser.to_dict()
        if self.scene is not None:
            data["scene"] = self.scene.to_dict()
        return data


def load_config(path: str | Path) -> SynthCalConfig:
    """Load a v1 config YAML from disk."""

    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Failed to read config: {path}") from exc
    except yaml.YAMLError as exc:  # pragma: no cover (hard to trigger deterministically)
        raise ConfigError(f"Invalid YAML in config: {path}") from exc
    return SynthCalConfig.from_dict(data)


def save_config(config: SynthCalConfig, path: str | Path) -> None:
    """Write a v1 config YAML to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(
        config.to_dict(),
        sort_keys=False,
        indent=2,
        default_flow_style=False,
    )
    path.write_text(text, encoding="utf-8")
