from __future__ import annotations

import numpy as np

from synthcal.laser.model import normalize_plane, transform_plane


def test_normalize_plane_scales_d() -> None:
    plane = np.array([2.0, 0.0, 0.0, 4.0], dtype=np.float64)
    out = normalize_plane(plane)
    assert np.allclose(out, np.array([1.0, 0.0, 0.0, 2.0], dtype=np.float64), atol=1e-12, rtol=0.0)


def test_transform_plane_identity_is_noop() -> None:
    plane = np.array([0.1, -0.2, 0.3, 4.5], dtype=np.float64)
    out = transform_plane(np.eye(4, dtype=np.float64), plane)
    assert np.allclose(out, plane, atol=1e-12, rtol=0.0)


def test_transform_plane_translation_along_z_updates_d() -> None:
    T = np.eye(4, dtype=np.float64)
    T[2, 3] = 10.0

    plane = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)  # z=0 in src
    out = transform_plane(T, plane)
    assert np.allclose(
        out, np.array([0.0, 0.0, 1.0, -10.0], dtype=np.float64), atol=1e-12, rtol=0.0
    )


def test_transform_plane_rotation_updates_normal() -> None:
    # Rotate points from src -> dst by +90deg about Y.
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)

    plane = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)  # z=0 in src
    out = transform_plane(T, plane)
    assert np.allclose(out, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), atol=1e-12, rtol=0.0)
