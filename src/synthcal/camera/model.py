"""Pinhole camera model with OpenCV-style distortion (no OpenCV dependency).

This module implements:
- normalized <-> pixel conversions using `K`
- OpenCV radial+tangential distortion
- a simple fixed-point undistortion iteration

All computations use float64 for numerical stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_float64_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _scalarize(value: np.ndarray) -> float | np.ndarray:
    """Return a Python float when `value` is scalar-like, otherwise return the array."""

    if np.ndim(value) == 0:
        return float(value)
    return value


@dataclass(frozen=True)
class PinholeCamera:
    """Pinhole camera with OpenCV-style radial+tangential distortion.

    Parameters
    ----------
    resolution:
        `(width_px, height_px)`.
    K:
        3x3 camera intrinsic matrix.
    dist:
        Distortion coefficients `[k1, k2, k3, p1, p2]` following OpenCV conventions.
    """

    resolution: tuple[int, int]
    K: np.ndarray
    dist: np.ndarray

    def __post_init__(self) -> None:
        width, height = self.resolution
        if not (isinstance(width, int) and isinstance(height, int)):
            raise TypeError("resolution must be (int width, int height)")
        if width <= 0 or height <= 0:
            raise ValueError("resolution values must be > 0")

        K = _as_float64_array(self.K)
        if K.shape != (3, 3):
            raise ValueError(f"K must have shape (3, 3), got {K.shape}")

        dist = _as_float64_array(self.dist).reshape(-1)
        if dist.shape != (5,):
            raise ValueError(f"dist must have shape (5,), got {dist.shape}")

        object.__setattr__(self, "K", K)
        object.__setattr__(self, "dist", dist)

    def pixel_to_normalized(self, u_px: float | np.ndarray, v_px: float | np.ndarray) -> tuple[Any, Any]:
        """Convert pixel coordinates to normalized coordinates using `K^{-1}`."""

        u = _as_float64_array(u_px)
        v = _as_float64_array(v_px)
        u, v = np.broadcast_arrays(u, v)
        uv1 = np.stack([u, v, np.ones_like(u)], axis=0)

        xy1 = np.linalg.inv(self.K) @ uv1
        x = xy1[0] / xy1[2]
        y = xy1[1] / xy1[2]
        return _scalarize(x), _scalarize(y)

    def normalized_to_pixel(self, x: float | np.ndarray, y: float | np.ndarray) -> tuple[Any, Any]:
        """Convert normalized coordinates to pixel coordinates using `K`."""

        x = _as_float64_array(x)
        y = _as_float64_array(y)
        x, y = np.broadcast_arrays(x, y)
        xy1 = np.stack([x, y, np.ones_like(x)], axis=0)
        uv1 = self.K @ xy1
        u = uv1[0] / uv1[2]
        v = uv1[1] / uv1[2]
        return _scalarize(u), _scalarize(v)

    def distort_normalized(self, xu: float | np.ndarray, yu: float | np.ndarray) -> tuple[Any, Any]:
        """Apply OpenCV radial+tangential distortion to normalized coordinates.

        Formula (OpenCV, normalized):
            r2 = x^2 + y^2
            radial = 1 + k1*r2 + k2*r2^2 + k3*r2^3
            x_tan = 2*p1*x*y + p2*(r2 + 2*x^2)
            y_tan = p1*(r2 + 2*y^2) + 2*p2*x*y
            xd = x*radial + x_tan
            yd = y*radial + y_tan
        """

        x = _as_float64_array(xu)
        y = _as_float64_array(yu)
        x, y = np.broadcast_arrays(x, y)
        k1, k2, k3, p1, p2 = (float(v) for v in self.dist)

        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

        x_tan = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_tan = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

        xd = x * radial + x_tan
        yd = y * radial + y_tan
        return _scalarize(xd), _scalarize(yd)

    def undistort_normalized(
        self,
        xd: float | np.ndarray,
        yd: float | np.ndarray,
        *,
        max_iters: int = 10,
        eps: float = 1e-12,
    ) -> tuple[Any, Any]:
        """Invert distortion via fixed-point iteration.

        Notes
        -----
        This is a simple method that works well for small/moderate distortion and
        points not too far from the principal point. It may converge slowly or
        fail to converge for extreme distortion / wide FOV.
        """

        if max_iters <= 0:
            raise ValueError("max_iters must be > 0")
        if eps <= 0:
            raise ValueError("eps must be > 0")

        xd_arr = _as_float64_array(xd)
        yd_arr = _as_float64_array(yd)
        xd_arr, yd_arr = np.broadcast_arrays(xd_arr, yd_arr)

        xu = xd_arr.copy()
        yu = yd_arr.copy()

        for _ in range(max_iters):
            x_hat, y_hat = self.distort_normalized(xu, yu)
            x_hat = _as_float64_array(x_hat)
            y_hat = _as_float64_array(y_hat)

            err_x = x_hat - xd_arr
            err_y = y_hat - yd_arr
            err_norm = np.sqrt(err_x * err_x + err_y * err_y)

            xu = xu - err_x
            yu = yu - err_y

            if bool(np.all(err_norm < eps)):
                break

        return _scalarize(xu), _scalarize(yu)

    def project_point_cam(self, X: np.ndarray) -> tuple[float, float, bool]:
        """Project a 3D point in the camera frame to pixels.

        Returns
        -------
        u_px, v_px, behind:
            `behind` is True when `Z <= 0` (point on/behind the camera plane). In that
            case, `(u_px, v_px)` are returned as NaN.
        """

        X = _as_float64_array(X).reshape(-1)
        if X.shape != (3,):
            raise ValueError(f"X must have shape (3,), got {X.shape}")

        z = float(X[2])
        if z <= 0.0:
            return float("nan"), float("nan"), True

        x = float(X[0] / z)
        y = float(X[1] / z)
        xd, yd = self.distort_normalized(x, y)
        u, v = self.normalized_to_pixel(xd, yd)
        return float(u), float(v), False

