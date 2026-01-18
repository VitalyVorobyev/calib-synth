from __future__ import annotations

import numpy as np

from synthcal.camera import PinholeCamera
from synthcal.render.chessboard import render_chessboard_image
from synthcal.render.gt import project_corners_px
from synthcal.targets.chessboard import ChessboardTarget


def test_chessboard_render_and_corner_projection() -> None:
    cam = PinholeCamera(
        resolution=(640, 480),
        K=np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]),
        dist=np.zeros(5, dtype=np.float64),
    )
    target = ChessboardTarget(inner_rows=6, inner_cols=9, square_size_mm=25.0)

    width_mm, height_mm = target.bounds()
    T_cam_target = np.eye(4, dtype=np.float64)
    T_cam_target[:3, 3] = np.array([-width_mm / 2.0, -height_mm / 2.0, 1000.0], dtype=np.float64)

    img = render_chessboard_image(cam, target, T_cam_target, background=128)
    assert img.dtype == np.uint8
    assert img.shape == (480, 640)
    assert np.unique(img).size >= 2

    corners_xyz = target.corners_xyz()
    corners_px, visible = project_corners_px(cam, corners_xyz, T_cam_target)
    assert corners_px.shape == (target.num_corners, 2)
    assert corners_px.dtype == np.float32
    assert visible.shape == (target.num_corners,)
    assert visible.dtype == bool
    assert int(np.sum(visible)) >= target.num_corners // 2

    u = corners_px[visible, 0]
    v = corners_px[visible, 1]
    assert np.all(u >= 0.0) and np.all(u < cam.resolution[0])
    assert np.all(v >= 0.0) and np.all(v < cam.resolution[1])
