# synthcal

Synthetic dataset generator for **camera + laser-stripe calibration**.

## Scope (v0)

This repository focuses on:
- Generating datasets of **static robot poses** for an **eye-in-hand multi-camera rig**.
- For each frame and camera, producing paired outputs from the **same pose**:
  1) `target.png`: chessboard under normal illumination
  2) `stripe.png`: black background with only a laser stripe (when laser is enabled)
  3) `corners_px.npy` (`float32`, `N x 2`) + `corners_visible.npy` (`bool`, `N`)
  4) `stripe_centerline_px.npy` (`float32`, `M x 2`) + `stripe_centerline_visible.npy` (`bool`, `M`)

**Units:** millimeters (mm) everywhere for geometry. Pixel coordinates are in pixels.

**Camera model:** OpenCV-style intrinsics `K` + distortion `dist = (k1, k2, k3, p1, p2)` (no OpenCV dependency).

## Camera model

Projection uses the standard pinhole model with distortion applied in **normalized** coordinates `(x, y)`:

```
r2 = x^2 + y^2
radial = 1 + k1*r2 + k2*r2^2 + k3*r2^3
x_tan = 2*p1*x*y + p2*(r2 + 2*x^2)
y_tan = p1*(r2 + 2*y^2) + 2*p2*x*y
xd = x*radial + x_tan
yd = y*radial + y_tan
```

Undistortion is implemented as a simple fixed-point iteration starting from `(xu, yu) = (xd, yd)` and repeatedly subtracting the forward-model error until convergence (works well for small/moderate distortion and points near the principal point).

## Current state

The `generate` subcommand currently writes:
- `config.yaml` (the normalized config used to generate)
- `manifest.yaml` (stable schema v1)
- per-frame `T_base_tcp.npy` (currently identity for all frames, v0)
- per-frame/per-camera `*_target.png` + `*_corners_*.npy`
- when laser is enabled: per-frame/per-camera `*_stripe.png` + `*_stripe_centerline_*.npy`
- placeholder rig/camera YAML files (`rig/`)

## CLI

Initialize an example config:

```bash
python -m synthcal init-config config.yaml
```

Create an output directory with manifest + placeholders:

```bash
python -m synthcal generate config.yaml out_dataset/
```

Preview one frame/camera with a matplotlib overlay:

```bash
python -m synthcal preview config.yaml --frame 0 --cam cam00
```

## Output format (planned)

The dataset layout is described in `manifest.yaml`. The v1 layout patterns include:
- `frames/frame_{frame_index:06d}/{camera_name}_target.png`
- `frames/frame_{frame_index:06d}/{camera_name}_corners_px.npy`
- `frames/frame_{frame_index:06d}/{camera_name}_corners_visible.npy`
- when laser is enabled:
  - `frames/frame_{frame_index:06d}/{camera_name}_stripe.png`
  - `frames/frame_{frame_index:06d}/{camera_name}_stripe_centerline_px.npy`
  - `frames/frame_{frame_index:06d}/{camera_name}_stripe_centerline_visible.npy`

Additional files created per frame (not currently described by the manifest):
- `frames/frame_{frame_index:06d}/T_base_tcp.npy`

## Coordinate conventions

- `T_cam_target` maps target-frame points into the camera frame: `X_cam = T_cam_target @ [X_target, 1]`.
- The chessboard target lies in plane `Z=0` in the target frame, with outer corner at `(0,0,0)`.
- Inner corners are ordered row-major (rows first, then cols), matching OpenCV’s convention.
