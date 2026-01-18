"""Config schema (v1) and YAML helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from synthcal.util import require_ascii_filename_component


class ConfigError(ValueError):
    """Raised when a config file is invalid or cannot be parsed."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping/object")
    return value


def _require_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    return value


def _require_number(value: Any, *, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number")
    return float(value)


def _require_seq(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ConfigError(f"{label} must be a sequence/list")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
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
        if cols <= 0 or rows <= 0:
            raise ConfigError("chessboard.inner_corners must be > 0")
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


@dataclass(frozen=True)
class SynthCalConfig:
    """Top-level config file."""

    version: int
    seed: int
    dataset: DatasetConfig
    rig: RigConfig
    chessboard: ChessboardConfig
    laser: LaserConfig | None
    scene: SceneConfig | None

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
        width_mm = (chessboard.inner_corners[0] + 1) * chessboard.square_size_mm
        height_mm = (chessboard.inner_corners[1] + 1) * chessboard.square_size_mm

        return cls(
            version=1,
            seed=0,
            dataset=DatasetConfig(name="example_dataset", num_frames=1),
            rig=RigConfig(cameras=(cam0, cam1)),
            chessboard=chessboard,
            laser=None,
            scene=SceneConfig(
                T_world_target=(
                    (1.0, 0.0, 0.0, -width_mm / 2.0),
                    (0.0, 1.0, 0.0, -height_mm / 2.0),
                    (0.0, 0.0, 1.0, 1000.0),
                    (0.0, 0.0, 0.0, 1.0),
                )
            ),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SynthCalConfig:
        data = _require_mapping(data, label="config")
        version = _require_int(data.get("version"), label="version")
        if version != 1:
            raise ConfigError(f"Unsupported config version {version}; expected 1")
        seed = _require_int(data.get("seed"), label="seed")
        if seed < 0:
            raise ConfigError("seed must be >= 0")
        dataset = DatasetConfig.from_dict(data.get("dataset"))
        rig = RigConfig.from_dict(data.get("rig"))
        chessboard = ChessboardConfig.from_dict(data.get("chessboard"))
        laser = LaserConfig.from_optional(data.get("laser"))
        if laser is not None and laser.enabled is False:
            laser = None
        scene = SceneConfig.from_optional(data.get("scene"))
        return cls(
            version=version,
            seed=seed,
            dataset=dataset,
            rig=rig,
            chessboard=chessboard,
            laser=laser,
            scene=scene,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "version": self.version,
            "seed": self.seed,
            "dataset": self.dataset.to_dict(),
            "rig": self.rig.to_dict(),
            "chessboard": self.chessboard.to_dict(),
        }
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
