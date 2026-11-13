import numpy as np
import pytest
from scipy import ndimage

from hycom_emulator.blur_budget import (
    alias_share, band_mask, band_slope, band_sums, bootstrap_slope, ensemble_limit, hann_taper, slope_stats,
)
from hycom_emulator.rollout_rea import BANDS, band_index

N, DX = 96, 8.0  # 96 x 96 cells of 8 km, the pack's stride-2 spacing
STARTS = 64


@pytest.fixture(scope="module")
def fields():
    """Truth = predictable part + an independent unpredictable part, over STARTS samples, with weights from a
    ragged region mask times a Hann window and the 50-100 km band."""
    rng = np.random.default_rng(1)
    predictable = rng.normal(size=(STARTS, N, N))
    unpredictable = 0.5 * rng.normal(size=(STARTS, N, N))
    mask = np.ones((N, N), bool)
    mask[:20, :30] = False  # land
    band = band_mask(band_index((N, N), DX), "50to100km")
    return predictable, predictable + unpredictable, hann_taper(mask), band, rng


def test_predictable_part_has_slope_one(fields):
    p, y, w, band, _ = fields
    assert band_slope(p, y, w, band)["slope"] == pytest.approx(1.0, abs=0.02)


def test_damped_forecast_has_slope_one_over_damping(fields):
    p, y, w, band, _ = fields
    assert band_slope(0.7 * p, y, w, band)["slope"] == pytest.approx(1 / 0.7, rel=0.02)


def test_added_noise_gives_slope_below_one(fields):
    p, y, w, band, rng = fields
    s = band_slope(p + 0.6 * rng.normal(size=p.shape), y, w, band)
    assert s["slope"] < 0.8  # expected 1 / (1 + 0.36)
    assert s["var_ratio"] > 1


def test_slope_is_corr_over_root_variance_ratio(fields):
    p, y, w, band, rng = fields
    for f in (p, 0.7 * p, p + rng.normal(size=p.shape), y[::-1]):
        s = band_slope(f, y, w, band)
        assert s["slope"] == pytest.approx(s["corr"] / np.sqrt(s["var_ratio"]), rel=1e-12)
        assert -1 <= s["corr"] <= 1


def test_band_sums_pool_by_summation_and_bootstrap_brackets_the_slope(fields):
    p, y, w, band, _ = fields
    sums = band_sums(0.8 * p, y, w, band)
    assert all(s.shape == (STARTS,) for s in sums)
    pooled = slope_stats(*sums)
    assert pooled == band_slope(0.8 * p, y, w, band)
    lo, hi = bootstrap_slope(*sums, n=500)
    assert lo < pooled["slope"] < hi
    assert hi - lo < 0.1


def test_gt100km_is_the_union_of_the_large_scale_bands():
    index = band_index((N, N), DX)
    names = list(BANDS)
    gt = band_mask(index, "gt100km")
    assert gt.sum() == ((index == names.index("100to200km")) | (index == names.index("gt200km"))).sum() > 0
    assert not (gt & band_mask(index, "50to100km")).any()
    assert not gt[0, 0]  # the mean is in no band


def test_band_sums_follow_rollout_band_variance_normalisation():
    """With no mask and no taper the four rollout_rea bands sum to the variance (Parseval)."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(N, N))
    w = np.ones((N, N))
    index = band_index((N, N), DX)
    total = sum(band_sums(x, x, w, index == i)[0] for i in range(len(BANDS)))
    assert total == pytest.approx(x.var(), rel=1e-10)


def test_power_above_the_subsample_nyquist_folds_entirely():
    """A wave of 3 cells (12 km at 4 km) is above the stride-2 Nyquist (16 km); subsampled it reads as 24 km,
    inside lt50km, where the full field holds nothing."""
    n, dx = 128, 4.0
    rng = np.random.default_rng(3)
    x = np.arange(n)
    waves = np.stack([np.cos(2 * np.pi * x / 3 + ph)[None, :].repeat(n, 0) for ph in rng.uniform(0, 2 * np.pi, 6)])
    share = alias_share(waves, 2, "lt50km", dx)
    assert share.shape == (6,)
    assert np.all(share > 0.95)


def test_smooth_field_folds_nothing():
    n, dx = 128, 4.0
    rng = np.random.default_rng(4)
    smooth = np.stack([ndimage.gaussian_filter(rng.normal(size=(n, n)), 4, mode="wrap") for _ in range(6)])
    smooth[:, :10, :25] = np.nan  # land, found from the NaNs
    for band in ("lt50km", "50to100km"):
        assert np.abs(alias_share(smooth, 2, band, dx)).max() < 0.03


def test_ensemble_limit_recovers_the_distribution_mean_error():
    """Truth N(0, 1), members N(0, 0.5) independent of it: the M-member mean has MSE 1 + 0.5 / M."""
    rng = np.random.default_rng(5)
    truth = rng.normal(size=200_000)
    members = rng.normal(scale=np.sqrt(0.5), size=(32, truth.size))
    mse = {m: float(((members[:m].mean(0) - truth) ** 2).mean()) for m in (4, 8, 16, 32)}
    assert ensemble_limit(mse) == pytest.approx(1.0, abs=0.02)
    assert ensemble_limit({8: 1 + 0.5 / 8, 16: 1 + 0.5 / 16, 32: 1 + 0.5 / 32}) == pytest.approx(1.0, abs=1e-12)
