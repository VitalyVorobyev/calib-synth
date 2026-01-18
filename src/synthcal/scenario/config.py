"""Scenario configuration models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RangeConfig:
    """Inclusive numeric range."""

    min: float
    max: float

    def to_dict(self) -> dict[str, Any]:
        return {"min": float(self.min), "max": float(self.max)}


@dataclass(frozen=True)
class InViewConfig:
    """Visibility constraints for a sampled pose."""

    margin_px: int = 20
    require_all_cameras: bool = True
    min_cameras_visible: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "margin_px": int(self.margin_px),
            "require_all_cameras": bool(self.require_all_cameras),
            "min_cameras_visible": int(self.min_cameras_visible),
        }


@dataclass(frozen=True)
class ScenarioConfig:
    """Pose sampling configuration.

    All distances are in millimeters (mm). Angles are in degrees.
    """

    num_frames: int | None
    distance_mm: RangeConfig
    tilt_deg: RangeConfig
    yaw_deg: RangeConfig
    roll_deg: RangeConfig
    xy_offset_frac: RangeConfig
    in_view: InViewConfig
    max_attempts_per_frame: int
    preset: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "num_frames": None if self.num_frames is None else int(self.num_frames),
            "distance_mm": self.distance_mm.to_dict(),
            "tilt_deg": self.tilt_deg.to_dict(),
            "yaw_deg": self.yaw_deg.to_dict(),
            "roll_deg": self.roll_deg.to_dict(),
            "xy_offset_frac": self.xy_offset_frac.to_dict(),
            "in_view": self.in_view.to_dict(),
            "max_attempts_per_frame": int(self.max_attempts_per_frame),
        }
        if self.preset is not None:
            data["preset"] = self.preset
        return data
