# Manifest reference (v1)

The manifest is a stable description of a generated dataset. Paths stored in the manifest are
relative to the dataset root directory.

Top-level keys:

- `version` (int): schema version (currently `1`).
- `manifest_version` (int): schema version (currently `1`, kept for backward compatibility).
- `created_utc` (str): ISO-8601 UTC timestamp.
- `generator` (object): `{name, version}`.
- `seed` (int): global dataset seed used.
- `units` (object): `{length: "mm"}`.
- `dataset` (object): `{name, num_frames}`.
- `paths` (object): key dataset paths (`config_yaml`, `manifest_yaml`, `rig_yaml`, `cameras_dir`, `frames_dir`).
- `cameras` (list): per-camera metadata (`name`, `intrinsics_yaml`, `image_size_px`).
- `layout` (object): filename patterns for per-frame/per-camera outputs.
- `laser` (object, optional): present only when laser outputs were generated.

