from __future__ import annotations

import numpy as np

from synthcal.camera import PinholeCamera
from synthcal.core.geometry import invert_se3
from synthcal.laser import intersect_planes_to_line, sample_line_points, transform_plane
from synthcal.render.gt import project_corners_px
from synthcal.render.stripe import render_stripe_image
from synthcal.targets.chessboard import ChessboardTarget


def _make_test_scene() -> tuple[PinholeCamera, ChessboardTarget, np.ndarray]:
    cam = PinholeCamera(
        resolution=(640, 480),
        K=np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]),
        dist=np.zeros(5, dtype=np.float64),
    )
    target = ChessboardTarget(inner_rows=6, inner_cols=9, square_size_mm=25.0)

    width_mm, height_mm = target.bounds()
    T_cam_target = np.eye(4, dtype=np.float64)
    T_cam_target[:3, 3] = np.array([-width_mm / 2.0, -height_mm / 2.0, 1000.0], dtype=np.float64)
    return cam, target, T_cam_target


def _compute_stripe(
    cam: PinholeCamera,
    target: ChessboardTarget,
    T_cam_target: np.ndarray,
    plane_in_tcp: np.ndarray,
    T_tcp_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    board_plane_target = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)

    T_cam_tcp = invert_se3(T_tcp_cam)
    plane_cam = transform_plane(T_cam_tcp, plane_in_tcp)
    T_target_cam = invert_se3(T_cam_target)
    plane_target = transform_plane(T_target_cam, plane_cam)

    try:
        p0, v = intersect_planes_to_line(plane_target, board_plane_target, eps=1e-12)
    except ValueError:
        stripe_px = np.zeros((0, 2), dtype=np.float32)
        stripe_vis = np.zeros((0,), dtype=bool)
        stripe_img = np.zeros((cam.resolution[1], cam.resolution[0]), dtype=np.uint8)
        return stripe_px, stripe_vis, stripe_img

    width_mm, height_mm = target.bounds()
    L = float(max(width_mm, height_mm) * 4.0)
    num = int(max(cam.resolution) * 2)
    pts_target = sample_line_points(p0, v, -L, L, num)
    stripe_px, stripe_vis = project_corners_px(cam, pts_target, T_cam_target)
    stripe_img = render_stripe_image(
        cam, stripe_px, stripe_vis, width_px=3.0, intensity=255, background=0
    )
    return stripe_px, stripe_vis, stripe_img


def test_stripe_centerline_and_image_shapes() -> None:
    cam, target, T_cam_target = _make_test_scene()

    width_mm, _height_mm = target.bounds()
    plane_target = np.array([1.0, 0.0, 0.0, -width_mm / 2.0], dtype=np.float64)  # x = width/2
    plane_tcp = transform_plane(T_cam_target, plane_target)

    stripe_px, stripe_vis, stripe_img = _compute_stripe(
        cam,
        target,
        T_cam_target,
        plane_in_tcp=plane_tcp,
        T_tcp_cam=np.eye(4, dtype=np.float64),
    )

    M = int(max(cam.resolution) * 2)
    assert stripe_px.shape == (M, 2)
    assert stripe_px.dtype == np.float32
    assert stripe_vis.shape == (M,)
    assert stripe_vis.dtype == bool
    assert stripe_img.shape == (cam.resolution[1], cam.resolution[0])
    assert stripe_img.dtype == np.uint8
    assert bool(np.any(stripe_vis))
    assert int(stripe_img.max()) > 0


def test_parallel_planes_produce_black_and_empty() -> None:
    cam, target, T_cam_target = _make_test_scene()

    # A laser plane parallel to the target plane (z=0 in target) yields no stable intersection line.
    plane_target = np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float64)  # z = 1mm
    plane_tcp = transform_plane(T_cam_target, plane_target)

    stripe_px, stripe_vis, stripe_img = _compute_stripe(
        cam,
        target,
        T_cam_target,
        plane_in_tcp=plane_tcp,
        T_tcp_cam=np.eye(4, dtype=np.float64),
    )

    assert stripe_px.shape == (0, 2)
    assert stripe_vis.shape == (0,)
    assert stripe_img.shape == (cam.resolution[1], cam.resolution[0])
    assert int(stripe_img.max()) == 0
