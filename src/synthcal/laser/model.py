"""Laser plane model and coordinate transforms.

All geometry is expressed in millimeters (mm).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from synthcal.core.geometry import as_se3, invert_se3


@dataclass(frozen=True)
class LaserPlane:
    """Infinite laser plane described in the TCP frame.

    The plane uses the implicit equation:

        n · X + d = 0

    where `X` is a 3D point in millimeters (mm).
    """

    plane_in_tcp: np.ndarray
    enabled: bool = True

    def __post_init__(self) -> None:
        plane = np.asarray(self.plane_in_tcp, dtype=np.float64)
        if plane.shape != (4,):
            raise ValueError(f"plane_in_tcp must have shape (4,), got {plane.shape}")
        if self.enabled and float(np.linalg.norm(plane[:3])) <= 0.0:
            raise ValueError("plane_in_tcp normal must be non-zero when enabled=True")
        object.__setattr__(self, "plane_in_tcp", plane)


def normalize_plane(plane: Any) -> np.ndarray:
    """Return a normalized copy of a plane equation.

    Parameters
    ----------
    plane:
        Array-like shape (4,) representing `[n_x, n_y, n_z, d]` where the plane is
        `n·X + d = 0` (X in mm).

    Returns
    -------
    np.ndarray
        Float64 array of shape (4,) where the normal `n` has unit length.
    """

    p = np.asarray(plane, dtype=np.float64)
    if p.shape != (4,):
        raise ValueError(f"plane must have shape (4,), got {p.shape}")
    n = p[:3]
    norm = float(np.linalg.norm(n))
    if norm <= 0.0:
        raise ValueError("plane normal must be non-zero")
    return p / norm


def transform_plane(T_dst_src: Any, plane_src: Any) -> np.ndarray:
    """Transform a plane equation between coordinate frames.

    Uses column-vector convention for points:

        X_dst = T_dst_src @ [X_src, 1]

    Plane coefficients transform as:

        π_dst = (T_dst_src^{-1})^T π_src

    where π = [n_x, n_y, n_z, d] and the plane is `n·X + d = 0` (X in mm).

    Parameters
    ----------
    T_dst_src:
        SE(3) transform mapping points from `src` to `dst`, shape (4, 4).
    plane_src:
        Plane coefficients in `src`, shape (4,).

    Returns
    -------
    np.ndarray
        Plane coefficients in `dst`, shape (4,), float64.
    """

    T = as_se3(T_dst_src)
    p = np.asarray(plane_src, dtype=np.float64)
    if p.shape != (4,):
        raise ValueError(f"plane_src must have shape (4,), got {p.shape}")
    T_inv = invert_se3(T)
    return (T_inv.T @ p).astype(np.float64, copy=False)
