"""synthcal: synthetic dataset generator for camera + laser-stripe calibration."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

__all__ = [
    "__version__",
    "generate_dataset",
    "generate_dataset_from_config",
    "render_frame_preview",
]

try:
    __version__ = _dist_version("synthcal")
except PackageNotFoundError:  # pragma: no cover
    # Allow running from a source checkout without installation (e.g.
    # `PYTHONPATH=src python -m synthcal ...`).
    _root = Path(__file__).resolve().parents[2]
    _pyproject = _root / "pyproject.toml"
    _version = None
    if _pyproject.is_file():
        m = re.search(
            r"""(?m)^version\s*=\s*["'](?P<version>[^"']+)["']\s*$""",
            _pyproject.read_text(encoding="utf-8"),
        )
        if m is not None:
            _version = m.group("version")
    __version__ = _version or "0.0.0"

if TYPE_CHECKING:  # pragma: no cover
    from synthcal.io.config import SynthCalConfig


def generate_dataset(config_path: str | Path, out_dir: str | Path) -> Path:
    """Generate a dataset from a config file on disk."""

    from synthcal.api import generate_dataset as _generate_dataset

    return _generate_dataset(config_path, out_dir)


def generate_dataset_from_config(config_obj: SynthCalConfig, out_dir: str | Path) -> Path:
    """Generate a dataset from an in-memory config object."""

    from synthcal.api import generate_dataset_from_config as _generate_dataset_from_config

    return _generate_dataset_from_config(config_obj, out_dir)


def render_frame_preview(config_obj: SynthCalConfig, frame_id: int) -> dict[str, Any]:
    """Render one frame worth of images and geometric ground truth."""

    from synthcal.api import render_frame_preview as _render_frame_preview

    return _render_frame_preview(config_obj, frame_id)
