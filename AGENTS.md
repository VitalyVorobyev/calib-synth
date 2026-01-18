# Contributor guide (AGENTS)

This repo is an early-stage synthetic dataset generator. Please keep changes small, deterministic, and well-tested.

## Requirements for changes

- **Tests required:** any feature or bugfix must include/extend pytest tests.
- **Determinism:** all randomness must be driven by a **single global seed** in the config; manifest must store the seed used.
- **Units:** geometry uses **millimeters (mm)** everywhere. Be explicit about pixel vs mm in names/docstrings.
- **No OpenCV:** do not add an `opencv-python` dependency or `import cv2`.
- **ASCII filenames:** generated dataset filenames must be ASCII-only and stable.
- **Style:** keep code in `src/` formatted with `ruff format` and linted with `ruff check`.
- **Docs:** public functions/classes should have docstrings; keep `README.md` in sync with format changes.

## Development workflow

- Create a venv and install:
  - `pip install -e '.[dev]'`
- Run:
  - `python -m pytest`
  - `ruff format .`
  - `ruff check .`

## Manifest schema changes

The manifest schema is versioned (`manifest_version`). If you need to change it:
- prefer additive changes
- bump `manifest_version` only for breaking changes
- add/extend tests that load and validate the manifest
