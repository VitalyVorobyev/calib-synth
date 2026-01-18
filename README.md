# synthcal

Synthetic dataset generator for **camera + laser-stripe calibration**.

## Install

```bash
pip install synthcal
# Optional (recommended for faster Gaussian blur):
pip install "synthcal[scipy]"
```

## Quickstart

Initialize an example config:

```bash
synthcal init-config config.yaml
```

Preview one frame/camera:

```bash
synthcal preview config.yaml --frame 0 --cam cam00
```

Generate a dataset:

```bash
synthcal generate config.yaml out_dataset/
```

`python -m synthcal ...` is also supported.

## Scope

This repository focuses on:
- Generating datasets of **static robot poses** for an **eye-in-hand multi-camera rig**.
- For each frame and camera, producing paired outputs from the **same pose**:
  1) `target.png`: chessboard under normal illumination
  2) `stripe.png`: black background with only a laser stripe (when laser is enabled)
  3) `corners_px.npy` (`float32`, `N x 2`) + `corners_visible.npy` (`bool`, `N`)
  4) `stripe_centerline_px.npy` (`float32`, `M x 2`) + `stripe_centerline_visible.npy` (`bool`, `M`)

**Units:** millimeters (mm) everywhere for geometry. Pixel coordinates are in pixels.

**Camera model:** OpenCV-style intrinsics `K` + distortion `dist = (k1, k2, k3, p1, p2)` (no OpenCV dependency).

## Examples

See `examples/minimal_no_laser.yaml`, `examples/minimal_with_laser.yaml`, and `examples/multicam_medium.yaml`.

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

## Outputs

The dataset layout is described in `manifest.yaml` (schema v1). Outputs include:
- `config.yaml` (the normalized config used to generate)
- `manifest.yaml` (stable schema v1; also includes `version: 1`)
- per-frame `T_base_tcp.npy` (identity for all frames unless `scenario` pose sampling is enabled)
- per-frame/per-camera `*_target.png` + `*_corners_*.npy`
- when laser is enabled: per-frame/per-camera `*_stripe.png` + `*_stripe_centerline_*.npy`
- placeholder rig/camera YAML files (`rig/`)

## Effects

Rendered images can be post-processed to add simple realism:
- `blur_sigma_px`: Gaussian defocus blur sigma in **pixels**
- `noise_sigma`: zero-mean Gaussian read noise sigma in **intensity units** (0..255)

Order of operations: blur → noise → clamp → quantize to uint8.
Blur uses SciPy (`scipy.ndimage.gaussian_filter`) when available; otherwise synthcal falls back to a
deterministic NumPy implementation (slower).

Determinism: noise is seeded from the global dataset seed and a stable per-output key
`(frame_index, camera_name, modality)` so re-generating with the same config+seed produces identical
images. Effects are applied to both `*_target.png` and `*_stripe.png` (when present).

## Scenario

Optional `scenario` pose sampling can generate a different `T_base_tcp` per frame while enforcing
in-view constraints (all corners inside the image with a margin). Presets `easy|medium|hard` provide
reasonable default ranges; all randomness is derived from the global seed.

## Laser

Laser output is optional. When enabled, synthcal models the laser as a single infinite plane in the
TCP frame: `n·X + d = 0` (X in mm). The stripe is rendered only where the laser plane intersects the
target plane (`Z=0` in target frame). If the planes are nearly parallel, synthcal outputs a black
stripe image and empty centerline arrays for that (frame, camera).

## Reproducibility

- `seed` is the single global seed recorded in `config.yaml` and `manifest.yaml`.
- Noise/effects use deterministic per-output RNG streams derived from `(seed, frame_index, camera_name, modality)`.
- Scenario pose sampling uses a deterministic per-frame stream derived from `(seed, frame_index, "__scenario__", "pose")`.

## Public API

Supported library entry points live in `src/synthcal/api.py`:

```python
from synthcal import generate_dataset, render_frame_preview
```

## CLI

The CLI provides:
- `synthcal init-config`
- `synthcal preview`
- `synthcal generate`

Preview options:
- `--no-effects`: show the raw render (before blur/noise/quantize)
- `--all-cams`: show a grid for all cameras
- `--show-stripe`: with `--all-cams`, also show a stripe grid when laser is enabled

## Coordinate conventions

- Frames (v0): `world == base` (eye-in-hand only).
- `T_tcp_cam` maps TCP-frame points into the camera frame.
- `T_cam_target` maps target-frame points into the camera frame: `X_cam = T_cam_target @ [X_target, 1]`.
- The chessboard target lies in plane `Z=0` in the target frame, with outer corner at `(0,0,0)`.
- Inner corners are ordered row-major (rows first, then cols), matching OpenCV’s convention.

## Docs

- `docs/config_reference.md`
- `docs/manifest_reference.md`

## Extending synthcal

Targets, sensors, sampling, and effects are implemented in `src/synthcal/`. Only the functions in
`src/synthcal/api.py` are intended to be stable for external users.
