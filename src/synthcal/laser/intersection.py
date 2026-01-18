"""Plane/plane intersection helpers.

All geometry is expressed in millimeters (mm).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def intersect_planes_to_line(
    plane_a: Any,
    plane_b: Any,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect two non-parallel planes into a 3D line.

    Parameters
    ----------
    plane_a, plane_b:
        Array-like shape (4,) representing `[n_x, n_y, n_z, d]` where the plane is
        `n·X + d = 0` (X in mm).
    eps:
        Threshold for detecting parallel/degenerate planes via `||n1 x n2||`.

    Returns
    -------
    (p0, v)
        `p0` is a point on the intersection line (shape (3,)).
        `v` is a unit direction vector for the line (shape (3,)).

    Raises
    ------
    ValueError
        If the planes are parallel/degenerate.
    """

    p1 = np.asarray(plane_a, dtype=np.float64)
    p2 = np.asarray(plane_b, dtype=np.float64)
    if p1.shape != (4,) or p2.shape != (4,):
        raise ValueError(f"plane_a/plane_b must have shape (4,), got {p1.shape} and {p2.shape}")

    n1, d1 = p1[:3], float(p1[3])
    n2, d2 = p2[:3], float(p2[3])

    v = np.cross(n1, n2)
    v_norm = float(np.linalg.norm(v))
    if v_norm < eps:
        raise ValueError("Planes are parallel or degenerate")
    v_unit = v / v_norm

    # Choose coordinate to eliminate (set to 0) based on the largest component
    # of the intersection direction. This yields a 2x2 system with determinant
    # equal to that component (up to sign), improving numerical stability.
    k = int(np.argmax(np.abs(v)))
    idx = [0, 1, 2]
    idx.remove(k)
    i, j = idx[0], idx[1]

    A = np.array([[n1[i], n1[j]], [n2[i], n2[j]]], dtype=np.float64)
    b = -np.array([d1, d2], dtype=np.float64)

    det = float(np.linalg.det(A))
    if abs(det) < eps:
        raise ValueError("Degenerate plane intersection (ill-conditioned)")

    sol = np.linalg.solve(A, b)
    p0 = np.zeros(3, dtype=np.float64)
    p0[i] = float(sol[0])
    p0[j] = float(sol[1])
    p0[k] = 0.0

    return p0, v_unit
