"""Small utilities used across synthcal.

Keep this module dependency-free (stdlib only).
"""

from __future__ import annotations

import re

_ASCII_FILENAME_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def require_ascii_filename_component(value: str, *, label: str) -> str:
    """Validate `value` is safe for use in generated filenames.

    The project constraint is "ASCII-only filenames". In addition, we restrict the
    component to a conservative subset to avoid platform-specific surprises.
    """

    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    if not _ASCII_FILENAME_COMPONENT_RE.fullmatch(value):
        raise ValueError(
            f"{label} must match {_ASCII_FILENAME_COMPONENT_RE.pattern} (got {value!r})"
        )
    return value
