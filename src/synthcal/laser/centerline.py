"""Laser stripe centerline sampling helpers.

All geometry is expressed in millimeters (mm).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def sample_line_points(
    p0: Any,
    v: Any,
    t_min: float,
    t_max: float,
    num: int,
) -> np.ndarray:
    """Sample points along a 3D line.

    The line is parameterized as `P(t) = p0 + t * v`.

    Parameters
    ----------
    p0:
        Point on the line, shape (3,) in mm.
    v:
        Line direction, shape (3,) in mm/mm (unitless). `v` does not need to be unit length.
    t_min, t_max:
        Range of `t` values to sample (inclusive).
    num:
        Number of samples (>= 2).

    Returns
    -------
    np.ndarray
        Sampled points with shape `(num, 3)`, float64, in mm.
    """

    if num < 2:
        raise ValueError("num must be >= 2")

    p0 = np.asarray(p0, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if p0.shape != (3,) or v.shape != (3,):
        raise ValueError(f"p0 and v must have shape (3,), got {p0.shape} and {v.shape}")

    t = np.linspace(float(t_min), float(t_max), int(num), dtype=np.float64)
    return p0.reshape(1, 3) + t.reshape(-1, 1) * v.reshape(1, 3)
