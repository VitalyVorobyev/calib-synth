# Releasing synthcal

This project publishes to PyPI from Git tags via GitHub Actions.

## One-time setup (Trusted Publishing)

1) Create the PyPI project `synthcal` (or confirm you own it).
2) In PyPI → *Publishing* → *Trusted Publishers*, add a GitHub publisher:
   - Repository: `VitalyVorobyev/calib-synth`
   - Workflow: `.github/workflows/publish.yml`
   - Environment (recommended): `pypi`
3) In GitHub, create an Environment named `pypi` and (optionally) require manual approval.

Token fallback (not preferred): set repo secret `PYPI_API_TOKEN`. The workflow will use it if present.

## Release steps

1) Bump the version in `pyproject.toml` (`[project].version`).
2) Update `CHANGELOG.md`.
3) Run local checks:
   - `python scripts/release_check.py`
   - `ruff format --check .`
   - `ruff check .`
   - `python -m pytest`
   - `python -m build`
   - `python -m twine check dist/*`
4) Tag and push:
   - `git tag vX.Y.Z`
   - `git push origin vX.Y.Z`

Pushing the tag triggers `.github/workflows/publish.yml`, which:
- builds sdist/wheel
- runs `twine check`
- publishes to PyPI (Trusted Publishing by default)

## Post-release verification

- Confirm the GitHub Actions workflow succeeded.
- Check PyPI: `https://pypi.org/project/synthcal/`
- Install in a clean env: `pip install synthcal` (and optionally `pip install "synthcal[scipy]"`)
- Run the README quickstart.

