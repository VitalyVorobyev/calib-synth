from __future__ import annotations

import numpy as np

from synthcal.camera import PinholeCamera


def test_identity_distortion_is_noop() -> None:
    cam = PinholeCamera(
        resolution=(640, 480),
        K=np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]),
        dist=np.zeros(5, dtype=np.float64),
    )

    pts = np.array([[0.0, 0.0], [0.1, -0.2], [-0.05, 0.04], [0.2, 0.15]], dtype=np.float64)
    xd, yd = cam.distort_normalized(pts[:, 0], pts[:, 1])
    assert np.allclose(xd, pts[:, 0], atol=1e-12, rtol=0.0)
    assert np.allclose(yd, pts[:, 1], atol=1e-12, rtol=0.0)

    xu, yu = cam.undistort_normalized(xd, yd, max_iters=3, eps=1e-15)
    assert np.allclose(xu, pts[:, 0], atol=1e-12, rtol=0.0)
    assert np.allclose(yu, pts[:, 1], atol=1e-12, rtol=0.0)


def test_distort_then_undistort_round_trip() -> None:
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.2, 0.2, size=(200, 2)).astype(np.float64)

    cam = PinholeCamera(
        resolution=(640, 480),
        K=np.eye(3, dtype=np.float64),
        dist=np.array([0.1, -0.05, 0.01, 0.001, -0.001], dtype=np.float64),
    )

    xd, yd = cam.distort_normalized(pts[:, 0], pts[:, 1])
    xu, yu = cam.undistort_normalized(xd, yd, max_iters=25, eps=1e-12)

    assert np.allclose(xu, pts[:, 0], atol=1e-8, rtol=0.0)
    assert np.allclose(yu, pts[:, 1], atol=1e-8, rtol=0.0)


def test_pixel_normalized_round_trip() -> None:
    cam = PinholeCamera(
        resolution=(1280, 720),
        K=np.array([[800.0, 5.0, 640.0], [0.0, 820.0, 360.0], [0.0, 0.0, 1.0]]),
        dist=np.zeros(5, dtype=np.float64),
    )

    u0, v0 = 1000.5, 500.25
    x, y = cam.pixel_to_normalized(u0, v0)
    u1, v1 = cam.normalized_to_pixel(x, y)

    assert np.isclose(u1, u0, atol=1e-12, rtol=0.0)
    assert np.isclose(v1, v0, atol=1e-12, rtol=0.0)


def test_project_point_cam_behind_flag() -> None:
    cam = PinholeCamera(
        resolution=(640, 480),
        K=np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]),
        dist=np.zeros(5, dtype=np.float64),
    )

    u, v, behind = cam.project_point_cam(np.array([0.0, 0.0, 1.0]))
    assert behind is False
    assert np.isclose(u, 320.0, atol=1e-12, rtol=0.0)
    assert np.isclose(v, 240.0, atol=1e-12, rtol=0.0)

    u, v, behind = cam.project_point_cam(np.array([0.0, 0.0, 0.0]))
    assert behind is True
    assert np.isnan(u) and np.isnan(v)

    u, v, behind = cam.project_point_cam(np.array([0.0, 0.0, -1.0]))
    assert behind is True
    assert np.isnan(u) and np.isnan(v)
