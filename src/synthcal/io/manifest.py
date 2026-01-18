"""Manifest schema (v1) and YAML helpers.

The manifest is the authoritative, stable description of a generated dataset.
All paths stored in the manifest are *relative* to the dataset root directory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class ManifestError(ValueError):
    """Raised when a manifest file is invalid or cannot be parsed."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{label} must be a mapping/object")
    return value


def _require_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{label} must be an integer")
    return value


def _require_number(value: Any, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ManifestError(f"{label} must be a number")
    return float(value)


def _require_seq(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ManifestError(f"{label} must be a sequence/list")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{label} must be a bool")
    return value


@dataclass(frozen=True)
class ManifestGenerator:
    name: str
    version: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ManifestGenerator:
        data = _require_mapping(data, label="generator")
        name = data.get("name")
        version = data.get("version")
        if not isinstance(name, str) or not name:
            raise ManifestError("generator.name must be a non-empty string")
        if not isinstance(version, str) or not version:
            raise ManifestError("generator.version must be a non-empty string")
        return cls(name=name, version=version)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True)
class ManifestPaths:
    """Paths to key files/directories in the dataset root."""

    config_yaml: str
    manifest_yaml: str
    rig_yaml: str
    cameras_dir: str
    frames_dir: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ManifestPaths:
        data = _require_mapping(data, label="paths")
        fields = ["config_yaml", "manifest_yaml", "rig_yaml", "cameras_dir", "frames_dir"]
        values: dict[str, str] = {}
        for field in fields:
            value = data.get(field)
            if not isinstance(value, str) or not value:
                raise ManifestError(f"paths.{field} must be a non-empty string")
            values[field] = value
        return cls(**values)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_yaml": self.config_yaml,
            "manifest_yaml": self.manifest_yaml,
            "rig_yaml": self.rig_yaml,
            "cameras_dir": self.cameras_dir,
            "frames_dir": self.frames_dir,
        }


@dataclass(frozen=True)
class ManifestCamera:
    name: str
    intrinsics_yaml: str
    image_size_px: tuple[int, int]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ManifestCamera:
        data = _require_mapping(data, label="camera")
        name = data.get("name")
        intrinsics_yaml = data.get("intrinsics_yaml")
        size = data.get("image_size_px")
        if not isinstance(name, str) or not name:
            raise ManifestError("camera.name must be a non-empty string")
        if not isinstance(intrinsics_yaml, str) or not intrinsics_yaml:
            raise ManifestError(f"camera[{name}].intrinsics_yaml must be a non-empty string")
        size_seq = _require_seq(size, label=f"camera[{name}].image_size_px")
        if len(size_seq) != 2:
            raise ManifestError(f"camera[{name}].image_size_px must be [width, height]")
        width = _require_int(size_seq[0], label=f"camera[{name}].image_size_px[0]")
        height = _require_int(size_seq[1], label=f"camera[{name}].image_size_px[1]")
        return cls(name=name, intrinsics_yaml=intrinsics_yaml, image_size_px=(width, height))

    def to_dict(self) -> dict[str, Any]:
        width, height = self.image_size_px
        return {
            "name": self.name,
            "intrinsics_yaml": self.intrinsics_yaml,
            "image_size_px": [width, height],
        }


@dataclass(frozen=True)
class ManifestLaser:
    """Laser parameters used to generate stripe outputs (present only when enabled)."""

    enabled: bool
    plane_in_tcp: tuple[float, float, float, float]
    stripe_width_px: int
    stripe_intensity: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ManifestLaser:
        data = _require_mapping(data, label="laser")
        enabled = _require_bool(data.get("enabled"), label="laser.enabled")

        plane_seq = _require_seq(data.get("plane_in_tcp"), label="laser.plane_in_tcp")
        if len(plane_seq) != 4:
            raise ManifestError("laser.plane_in_tcp must have length 4: [a, b, c, d]")
        plane = tuple(
            _require_number(plane_seq[i], label=f"laser.plane_in_tcp[{i}]") for i in range(4)
        )

        stripe_width_px = _require_int(data.get("stripe_width_px"), label="laser.stripe_width_px")
        stripe_intensity = _require_int(
            data.get("stripe_intensity"), label="laser.stripe_intensity"
        )

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
class ManifestLayout:
    """Pattern strings that describe where per-frame/per-camera outputs will live."""

    frame_dir: str
    camera_dir: str
    target_image: str
    corners_px_npy: str
    corners_visible_npy: str
    stripe_image: str | None = None
    stripe_centerline_px_npy: str | None = None
    stripe_centerline_visible_npy: str | None = None

    @classmethod
    def v1_default(cls, *, include_laser: bool = False) -> ManifestLayout:
        frame_dir = "frames/frame_{frame_index:06d}"
        camera_dir = "."
        base = frame_dir
        return cls(
            frame_dir=frame_dir,
            camera_dir=camera_dir,
            target_image=f"{base}/{{camera_name}}_target.png",
            corners_px_npy=f"{base}/{{camera_name}}_corners_px.npy",
            corners_visible_npy=f"{base}/{{camera_name}}_corners_visible.npy",
            stripe_image=(f"{base}/{{camera_name}}_stripe.png" if include_laser else None),
            stripe_centerline_px_npy=(
                f"{base}/{{camera_name}}_stripe_centerline_px.npy" if include_laser else None
            ),
            stripe_centerline_visible_npy=(
                f"{base}/{{camera_name}}_stripe_centerline_visible.npy" if include_laser else None
            ),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ManifestLayout:
        data = _require_mapping(data, label="layout")
        required = [
            "frame_dir",
            "camera_dir",
            "target_image",
            "corners_px_npy",
            "corners_visible_npy",
        ]
        values: dict[str, str] = {}
        for key in required:
            value = data.get(key)
            if not isinstance(value, str) or not value:
                raise ManifestError(f"layout.{key} must be a non-empty string")
            values[key] = value

        stripe_image = data.get("stripe_image")
        if stripe_image is not None and (not isinstance(stripe_image, str) or not stripe_image):
            raise ManifestError("layout.stripe_image must be a non-empty string when present")

        stripe_centerline = data.get("stripe_centerline_px_npy")
        if stripe_centerline is not None and (
            not isinstance(stripe_centerline, str) or not stripe_centerline
        ):
            raise ManifestError(
                "layout.stripe_centerline_px_npy must be a non-empty string when present"
            )

        stripe_visible = data.get("stripe_centerline_visible_npy")
        if stripe_visible is not None and (
            not isinstance(stripe_visible, str) or not stripe_visible
        ):
            raise ManifestError(
                "layout.stripe_centerline_visible_npy must be a non-empty string when present"
            )

        return cls(
            **values,  # type: ignore[arg-type]
            stripe_image=stripe_image,
            stripe_centerline_px_npy=stripe_centerline,
            stripe_centerline_visible_npy=stripe_visible,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "frame_dir": self.frame_dir,
            "camera_dir": self.camera_dir,
            "target_image": self.target_image,
            "corners_px_npy": self.corners_px_npy,
            "corners_visible_npy": self.corners_visible_npy,
        }
        if self.stripe_image is not None:
            data["stripe_image"] = self.stripe_image
        if self.stripe_centerline_px_npy is not None:
            data["stripe_centerline_px_npy"] = self.stripe_centerline_px_npy
        if self.stripe_centerline_visible_npy is not None:
            data["stripe_centerline_visible_npy"] = self.stripe_centerline_visible_npy
        return data


@dataclass(frozen=True)
class SynthCalManifest:
    """Top-level manifest (v1)."""

    manifest_version: int
    created_utc: str
    generator: ManifestGenerator
    seed: int
    units: dict[str, Any]
    dataset: dict[str, Any]
    laser: ManifestLaser | None
    paths: ManifestPaths
    cameras: tuple[ManifestCamera, ...]
    layout: ManifestLayout

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SynthCalManifest:
        data = _require_mapping(data, label="manifest")
        version = _require_int(data.get("manifest_version"), label="manifest_version")
        if version != 1:
            raise ManifestError(f"Unsupported manifest_version {version}; expected 1")

        created_utc = data.get("created_utc")
        if not isinstance(created_utc, str) or not created_utc:
            raise ManifestError("created_utc must be a non-empty string")

        generator = ManifestGenerator.from_dict(data.get("generator"))
        seed = _require_int(data.get("seed"), label="seed")
        units = _require_mapping(data.get("units"), label="units")
        dataset = _require_mapping(data.get("dataset"), label="dataset")
        laser_raw = data.get("laser")
        laser = None if laser_raw is None else ManifestLaser.from_dict(laser_raw)
        paths = ManifestPaths.from_dict(data.get("paths"))

        cameras_raw = _require_seq(data.get("cameras"), label="cameras")
        cameras = tuple(ManifestCamera.from_dict(c) for c in cameras_raw)
        layout = ManifestLayout.from_dict(data.get("layout"))

        return cls(
            manifest_version=version,
            created_utc=created_utc,
            generator=generator,
            seed=seed,
            units=dict(units),
            dataset=dict(dataset),
            laser=laser,
            paths=paths,
            cameras=cameras,
            layout=layout,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "manifest_version": self.manifest_version,
            "created_utc": self.created_utc,
            "generator": self.generator.to_dict(),
            "seed": self.seed,
            "units": dict(self.units),
            "dataset": dict(self.dataset),
            "paths": self.paths.to_dict(),
            "cameras": [c.to_dict() for c in self.cameras],
            "layout": self.layout.to_dict(),
        }
        if self.laser is not None and self.laser.enabled:
            data["laser"] = self.laser.to_dict()
        return data


def utc_now_iso8601() -> str:
    """Return current UTC time as an ISO-8601 string with 'Z' suffix."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_manifest(path: str | Path) -> SynthCalManifest:
    """Load a v1 manifest YAML from disk."""

    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"Failed to read manifest: {path}") from exc
    except yaml.YAMLError as exc:  # pragma: no cover
        raise ManifestError(f"Invalid YAML in manifest: {path}") from exc
    return SynthCalManifest.from_dict(data)


def save_manifest(manifest: SynthCalManifest, path: str | Path) -> None:
    """Write a v1 manifest YAML to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(
        manifest.to_dict(),
        sort_keys=False,
        indent=2,
        default_flow_style=False,
    )
    path.write_text(text, encoding="utf-8")
