from __future__ import annotations

import numpy as np

from synthcal.camera import PinholeCamera
from synthcal.scenario.config import InViewConfig, RangeConfig, ScenarioConfig
from synthcal.scenario.sampling import sample_valid_T_base_tcp, sample_valid_T_tcp_target
from synthcal.targets.chessboard import ChessboardTarget


def test_pose_sampling_produces_visible_frames() -> None:
    cam = PinholeCamera(
        resolution=(640, 480),
        K=np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]),
        dist=np.zeros(5, dtype=np.float64),
    )
    cameras = {"cam00": cam}
    rig_extrinsics = {"cam00": np.eye(4, dtype=np.float64)}

    target = ChessboardTarget(inner_rows=6, inner_cols=9, square_size_mm=25.0)
    T_world_target = np.eye(4, dtype=np.float64)

    scenario = ScenarioConfig(
        num_frames=5,
        distance_mm=RangeConfig(min=800.0, max=1000.0),
        tilt_deg=RangeConfig(min=0.0, max=20.0),
        yaw_deg=RangeConfig(min=-180.0, max=180.0),
        roll_deg=RangeConfig(min=-180.0, max=180.0),
        xy_offset_frac=RangeConfig(min=-0.05, max=0.05),
        in_view=InViewConfig(margin_px=20, require_all_cameras=True, min_cameras_visible=1),
        max_attempts_per_frame=200,
        preset=None,
    )

    for frame_id in range(5):
        T_base_tcp, vis = sample_valid_T_base_tcp(
            global_seed=0,
            frame_id=frame_id,
            scenario=scenario,
            cameras=cameras,
            rig_extrinsics=rig_extrinsics,
            target=target,
            T_world_target=T_world_target,
            reference_camera="cam00",
        )
        assert T_base_tcp.shape == (4, 4)
        assert vis == {"cam00": True}


def test_target_pose_sampling_produces_visible_frames_and_varies() -> None:
    cam = PinholeCamera(
        resolution=(640, 480),
        K=np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]),
        dist=np.zeros(5, dtype=np.float64),
    )
    cameras = {"cam00": cam}
    rig_extrinsics = {"cam00": np.eye(4, dtype=np.float64)}

    target = ChessboardTarget(inner_rows=6, inner_cols=9, square_size_mm=25.0)

    sampling = ScenarioConfig(
        num_frames=5,
        distance_mm=RangeConfig(min=800.0, max=1000.0),
        tilt_deg=RangeConfig(min=0.0, max=20.0),
        yaw_deg=RangeConfig(min=-180.0, max=180.0),
        roll_deg=RangeConfig(min=-180.0, max=180.0),
        xy_offset_frac=RangeConfig(min=-0.05, max=0.05),
        in_view=InViewConfig(margin_px=20, require_all_cameras=True, min_cameras_visible=1),
        max_attempts_per_frame=200,
        preset=None,
    )

    T0, vis0 = sample_valid_T_tcp_target(
        global_seed=0,
        frame_id=0,
        sampling=sampling,
        cameras=cameras,
        rig_extrinsics=rig_extrinsics,
        target=target,
        reference_camera="cam00",
    )
    T1, vis1 = sample_valid_T_tcp_target(
        global_seed=0,
        frame_id=1,
        sampling=sampling,
        cameras=cameras,
        rig_extrinsics=rig_extrinsics,
        target=target,
        reference_camera="cam00",
    )
    assert T0.shape == (4, 4)
    assert T1.shape == (4, 4)
    assert vis0 == {"cam00": True}
    assert vis1 == {"cam00": True}
    assert not np.allclose(T0, T1)

    # Determinism: same (seed, frame_id) yields identical pose.
    T0b, _vis0b = sample_valid_T_tcp_target(
        global_seed=0,
        frame_id=0,
        sampling=sampling,
        cameras=cameras,
        rig_extrinsics=rig_extrinsics,
        target=target,
        reference_camera="cam00",
    )
    assert np.allclose(T0, T0b)


def test_target_pose_sampling_multi_camera_visibility() -> None:
    cam = PinholeCamera(
        resolution=(640, 480),
        K=np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]),
        dist=np.zeros(5, dtype=np.float64),
    )
    cameras = {"cam00": cam, "cam01": cam}
    rig_extrinsics = {
        "cam00": np.eye(4, dtype=np.float64),
        "cam01": np.array(
            [
                [1.0, 0.0, 0.0, 100.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    }

    target = ChessboardTarget(inner_rows=6, inner_cols=9, square_size_mm=25.0)

    sampling = ScenarioConfig(
        num_frames=1,
        distance_mm=RangeConfig(min=900.0, max=1200.0),
        tilt_deg=RangeConfig(min=0.0, max=20.0),
        yaw_deg=RangeConfig(min=-180.0, max=180.0),
        roll_deg=RangeConfig(min=-180.0, max=180.0),
        xy_offset_frac=RangeConfig(min=-0.05, max=0.05),
        in_view=InViewConfig(margin_px=20, require_all_cameras=True, min_cameras_visible=1),
        max_attempts_per_frame=500,
        preset=None,
    )

    _T_tcp_target, vis = sample_valid_T_tcp_target(
        global_seed=0,
        frame_id=0,
        sampling=sampling,
        cameras=cameras,
        rig_extrinsics=rig_extrinsics,
        target=target,
        reference_camera="cam00",
    )
    assert vis == {"cam00": True, "cam01": True}


def test_pose_sampling_raises_on_impossible_constraints() -> None:
    cam = PinholeCamera(
        resolution=(640, 480),
        K=np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]),
        dist=np.zeros(5, dtype=np.float64),
    )
    cameras = {"cam00": cam}
    rig_extrinsics = {"cam00": np.eye(4, dtype=np.float64)}

    target = ChessboardTarget(inner_rows=6, inner_cols=9, square_size_mm=25.0)
    T_world_target = np.eye(4, dtype=np.float64)

    scenario = ScenarioConfig(
        num_frames=1,
        distance_mm=RangeConfig(min=10.0, max=10.0),
        tilt_deg=RangeConfig(min=0.0, max=0.0),
        yaw_deg=RangeConfig(min=0.0, max=0.0),
        roll_deg=RangeConfig(min=0.0, max=0.0),
        xy_offset_frac=RangeConfig(min=0.0, max=0.0),
        in_view=InViewConfig(margin_px=20, require_all_cameras=True, min_cameras_visible=1),
        max_attempts_per_frame=10,
        preset=None,
    )

    try:
        sample_valid_T_base_tcp(
            global_seed=0,
            frame_id=0,
            scenario=scenario,
            cameras=cameras,
            rig_extrinsics=rig_extrinsics,
            target=target,
            T_world_target=T_world_target,
            reference_camera="cam00",
        )
    except ValueError as exc:
        msg = str(exc)
        assert "Failed to sample a valid pose" in msg
        assert "max_attempts_per_frame" in msg
    else:
        raise AssertionError("Expected ValueError for impossible constraints")
