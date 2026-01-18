"""Chessboard calibration target model.

Coordinate system (target frame)
-------------------------------
- The chessboard lies in the plane `Z=0`.
- The outer corner of the board is at `(0, 0, 0)`.
- `+X` increases to the right across columns.
- `+Y` increases downward across rows.

Inner corners follow OpenCV convention and are returned in row-major order:
for `j` in rows (0..inner_rows-1), for `i` in cols (0..inner_cols-1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChessboardTarget:
    """A planar chessboard target (inner corners only)."""

    inner_rows: int
    inner_cols: int
    square_size_mm: float
    border_squares: int = 0
    colors: tuple[int, int] = (0, 255)

    def __post_init__(self) -> None:
        if self.inner_rows <= 0 or self.inner_cols <= 0:
            raise ValueError("inner_rows and inner_cols must be > 0")
        if self.square_size_mm <= 0:
            raise ValueError("square_size_mm must be > 0")
        if self.border_squares < 0:
            raise ValueError("border_squares must be >= 0")
        if len(self.colors) != 2:
            raise ValueError("colors must be a pair (black, white)")
        for c in self.colors:
            if not isinstance(c, int) or not (0 <= c <= 255):
                raise ValueError("colors must be integers in [0, 255]")

    @property
    def num_corners(self) -> int:
        return self.inner_rows * self.inner_cols

    def bounds(self) -> tuple[float, float]:
        """Return `(width_mm, height_mm)` of the board (including optional border)."""

        cols_squares = (self.inner_cols + 1) + 2 * self.border_squares
        rows_squares = (self.inner_rows + 1) + 2 * self.border_squares
        return cols_squares * self.square_size_mm, rows_squares * self.square_size_mm

    def corners_xyz(self) -> np.ndarray:
        """Return inner corner locations in the target frame, shape `(N, 3)` float64."""

        s = float(self.square_size_mm)
        x0 = (self.border_squares + 1) * s
        y0 = (self.border_squares + 1) * s

        xs = x0 + np.arange(self.inner_cols, dtype=np.float64) * s
        ys = y0 + np.arange(self.inner_rows, dtype=np.float64) * s
        xx, yy = np.meshgrid(xs, ys)  # (rows, cols)
        corners = np.stack([xx.reshape(-1), yy.reshape(-1), np.zeros(self.num_corners)], axis=1)
        return corners.astype(np.float64, copy=False)

    def eval_color_xy(self, x_mm: Any, y_mm: Any) -> np.ndarray:
        """Evaluate chessboard color at point(s) on the plane.

        Parameters
        ----------
        x_mm, y_mm:
            Coordinates in the target plane (mm). Scalars or arrays are accepted.

        Returns
        -------
        np.ndarray
            `uint8` values with the same broadcasted shape as inputs.
        """

        x = np.asarray(x_mm, dtype=np.float64)
        y = np.asarray(y_mm, dtype=np.float64)
        x, y = np.broadcast_arrays(x, y)

        s = float(self.square_size_mm)
        ix = np.floor(x / s).astype(np.int64)
        iy = np.floor(y / s).astype(np.int64)

        is_black = ((ix + iy) & 1) == 0
        black, white = self.colors
        return np.where(is_black, black, white).astype(np.uint8, copy=False)

