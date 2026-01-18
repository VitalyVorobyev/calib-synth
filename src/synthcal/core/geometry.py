"""Minimal SE(3) helpers.

This module intentionally stays small. As the project grows, extend it with
care and keep behavior well-tested.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def as_se3(T: Any, *, tol: float = 1e-9) -> np.ndarray:
    """Validate and normalize an SE(3) transform matrix.

    Parameters
    ----------
    T:
        Array-like 4x4 matrix.
    tol:
        Tolerance for validating the last row.

    Returns
    -------
    np.ndarray
        A float64 array with shape (4, 4). If the last row is close to
        `[0, 0, 0, 1]` within `tol`, it is snapped exactly to that value.
    """

    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Expected SE(3) matrix with shape (4, 4), got {T.shape}")

    expected_last = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if not np.allclose(T[3, :], expected_last, atol=tol, rtol=0.0):
        raise ValueError("Invalid SE(3) matrix: last row must be [0, 0, 0, 1]")

    Tn = T.copy()
    Tn[3, :] = expected_last
    return Tn


def transform_points(T: Any, points: Any) -> np.ndarray:
    """Apply an SE(3) transform to 3D point(s).

    Uses column-vector convention: `X_out = T @ [X_in, 1]`.

    Parameters
    ----------
    T:
        SE(3) transform, shape (4, 4).
    points:
        Point(s) as shape (3,) or (..., 3).

    Returns
    -------
    np.ndarray
        Transformed points with the same leading shape as `points`.
    """

    T = as_se3(T)
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape == (3,):
        return (T[:3, :3] @ pts) + T[:3, 3]
    if pts.ndim < 1 or pts.shape[-1] != 3:
        raise ValueError(f"points must have shape (3,) or (..., 3), got {pts.shape}")
    return pts @ T[:3, :3].T + T[:3, 3]


def invert_se3(T: Any) -> np.ndarray:
    """Invert an SE(3) transform.

    Uses the SE(3) structure:
        T = [[R, t],
             [0, 1]]
        T^{-1} = [[R^T, -R^T t],
                  [0,    1]]
    """

    T = as_se3(T)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv
