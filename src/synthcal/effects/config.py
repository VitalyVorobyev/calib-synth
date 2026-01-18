"""Configuration models for image effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EffectsConfig:
    """Effects stack applied to rendered images before saving.

    Order: blur -> noise -> clamp -> quantize (uint8).
    """

    enabled: bool = True
    blur_sigma_px: float = 0.0
    noise_sigma: float = 0.0
    clamp_min: float = 0.0
    clamp_max: float = 255.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "blur_sigma_px": float(self.blur_sigma_px),
            "noise_sigma": float(self.noise_sigma),
            "clamp_min": float(self.clamp_min),
            "clamp_max": float(self.clamp_max),
        }
