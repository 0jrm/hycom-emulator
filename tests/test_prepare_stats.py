"""B00 normalization statistics: stable moments and the diff-std floor."""

import numpy as np

from hycom_emulator.prepare_b00 import _Moments


def test_moments_match_numpy_across_batches():
    rng = np.random.default_rng(0)
    a = rng.normal(3e4, 50.0, size=(5000, 3)).astype(np.float32)
    acc = _Moments()
    for chunk in np.array_split(a, 7):
        acc.add(chunk)
    np.testing.assert_allclose(acc.mean, a.astype(np.float64).mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(acc.std, a.astype(np.float64).std(axis=0), rtol=1e-6)


def test_constant_feature_has_no_cancellation_residue():
    acc = _Moments()
    for _ in range(4):
        acc.add(np.full((1000, 1), 31771.439453125, dtype=np.float32))
    assert acc.std[0] == 1.0  # exact 0 -> _safe -> 1.0, not ~1e-13
