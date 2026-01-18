from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    # Allow `import synthcal` without requiring an editable install.
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.insert(0, str(src))
