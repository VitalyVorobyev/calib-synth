"""Image effects pipeline (blur, noise, quantization)."""

from __future__ import annotations

__all__ = ["EffectsConfig", "apply_effects"]

from .config import EffectsConfig
from .pipeline import apply_effects
