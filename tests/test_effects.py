from __future__ import annotations

import numpy as np

from synthcal.core.seeding import derive_rng
from synthcal.effects.config import EffectsConfig
from synthcal.effects.pipeline import apply_effects


def test_effects_noop_when_blur_and_noise_zero() -> None:
    img = (np.arange(64, dtype=np.uint8).reshape(8, 8) * 3) % 255
    cfg = EffectsConfig(enabled=True, blur_sigma_px=0.0, noise_sigma=0.0)
    out = apply_effects(img, cfg, rng=None)
    assert np.array_equal(out, img)


def test_effects_blur_spreads_impulse() -> None:
    img = np.zeros((21, 21), dtype=np.uint8)
    img[10, 10] = 255

    cfg = EffectsConfig(enabled=True, blur_sigma_px=1.0, noise_sigma=0.0)
    out = apply_effects(img, cfg, rng=None)

    assert out.dtype == np.uint8
    assert out.shape == img.shape
    assert int(np.count_nonzero(out)) > 1
    assert int(out.max()) < 255


def test_effects_noise_is_deterministic_per_modality() -> None:
    img = np.full((32, 32), 128, dtype=np.uint8)
    cfg = EffectsConfig(enabled=True, blur_sigma_px=0.0, noise_sigma=10.0)

    rng0 = derive_rng(123, 0, "cam00", "target")
    out0 = apply_effects(img, cfg, rng=rng0)
    rng1 = derive_rng(123, 0, "cam00", "target")
    out1 = apply_effects(img, cfg, rng=rng1)
    assert np.array_equal(out0, out1)

    rng2 = derive_rng(123, 0, "cam00", "stripe")
    out2 = apply_effects(img, cfg, rng=rng2)
    assert not np.array_equal(out0, out2)
