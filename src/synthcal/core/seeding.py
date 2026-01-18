"""Deterministic seeding utilities.

All randomness in synthcal should be derived from a single global seed stored in the
config/manifest. These helpers provide stable, cross-run seeding without relying on Python's
process-randomized `hash()`.
"""

from __future__ import annotations

import hashlib

import numpy as np


def stable_u64_from_str(s: str) -> int:
    """Return a stable uint64 derived from an input string.

    Uses SHA-256 over UTF-8 bytes and interprets the first 8 digest bytes as a little-endian
    unsigned integer.
    """

    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def derive_rng(global_seed: int, frame_id: int, cam_id: str, modality: str) -> np.random.Generator:
    """Derive a deterministic RNG for a specific (frame, camera, modality).

    Parameters
    ----------
    global_seed:
        Global dataset seed (from config/manifest).
    frame_id:
        Frame index.
    cam_id:
        Camera name/id (ASCII filename component).
    modality:
        Sub-stream label, e.g. `"target"` or `"stripe"`.
    """

    key = f"{int(global_seed)}|{int(frame_id)}|{cam_id}|{modality}"
    seed_u64 = stable_u64_from_str(key)
    return np.random.default_rng(seed_u64)
