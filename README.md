# synthcal

Synthetic dataset generator for **camera + laser-stripe calibration**.

## Scope (v0)

This repository focuses on:
- Generating datasets of **static robot poses** for an **eye-in-hand multi-camera rig**.
- For each frame and camera, producing paired outputs from the **same pose**:
  1) `target.png`: chessboard under normal illumination
  2) `stripe.png`: black background with only a laser stripe
  3) `corners_px.npy` (`float32`, `N x 2`) + `corners_visible.npy` (`bool`, `N`)
  4) `stripe_centerline_px.npy` (`float32`, `M x 2`)

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
- output folder structure
- `config.yaml` (the normalized config used to generate)
- `manifest.yaml` (stable schema v1)
- placeholder rig/camera YAML files

Rendering of images and ground-truth arrays will be added next.

## CLI

Initialize an example config:

```bash
python -m synthcal init-config config.yaml
```

Create an output directory with manifest + placeholders:

```bash
python -m synthcal generate config.yaml out_dataset/
```

## Output format (planned)

The dataset layout is described in `manifest.yaml`. The v1 layout patterns include:
- `frames/frame_{frame_index:06d}/cam_{camera_name}/target.png`
- `frames/frame_{frame_index:06d}/cam_{camera_name}/corners_px.npy`
- `frames/frame_{frame_index:06d}/cam_{camera_name}/corners_visible.npy`

When `laser.enabled: true` in the config, the manifest additionally lists:
- `frames/frame_{frame_index:06d}/cam_{camera_name}/stripe.png`
- `frames/frame_{frame_index:06d}/cam_{camera_name}/stripe_centerline_px.npy`

For v0, the generator creates the `frames/` folder and per-frame/per-camera directories but does not yet render the files.
