"""synthcal: synthetic dataset generator for camera + laser-stripe calibration."""

from __future__ import annotations

__all__ = [
    "__version__",
    "generate_dataset",
    "generate_dataset_from_config",
    "render_frame_preview",
]

__version__ = "0.1.0"

from .api import generate_dataset, generate_dataset_from_config, render_frame_preview  # noqa: E402
