"""The blur budget: how much of the variance a forecast loses is the share it cannot predict.

Pure functions for gom-da cards emu-rea-blur-budget and emu-rea-prob-rescore (plan 2026-11-13, Q1, Q3, Q4).

- band_slope: per wavelength band, the regression slope of the truth on the forecast through the origin,
  sum(f y) / sum(f f), with the variance ratio sum(f f) / sum(y y) and the correlation. The forecast that
  minimises squared error has slope 1; slope above 1 means damped predictable signal, below 1 added noise.
  slope = corr / sqrt(var_ratio) holds exactly.
- alias_share: the share of a subsampled field's power in a band that the full-resolution field does not hold
  at the same wavenumbers, i.e. power folded in from above the subsample's Nyquist.
- ensemble_limit: the intercept at 1/M = 0 of mean squared error against 1/M.

Bands are rollout_rea's (one definition, rollout_rea.band_index), with gt100km the union of its two
large-scale bands. Band sums follow rollout_rea.band_variance: remove the weighted mean, taper with the
weights (region mask x Hann window), FFT, and normalise so the bins sum to the tapered variance (Parseval).
"""

from __future__ import annotations

import numpy as np

from hycom_emulator.rollout_rea import BANDS, band_index

CARD_BANDS = {"lt50km": ("lt50km",), "50to100km": ("50to100km",), "gt100km": ("100to200km", "gt200km")}


def band_mask(index: np.ndarray, band: str) -> np.ndarray:
    """Boolean FFT-bin mask of a card band from a rollout_rea band index (Plane.band or band_index)."""
    names = list(BANDS)
    return np.isin(np.asarray(index), [names.index(b) for b in CARD_BANDS[band]])


def hann_taper(mask: np.ndarray) -> np.ndarray:
    """Region mask times a 2D Hann window, as rollout_rea.plane tapers its spectral box."""
    ny, nx = mask.shape
    return mask * np.outer(np.hanning(ny), np.hanning(nx))


def _spectrum(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """FFT of the tapered anomaly, scaled so that sum |X|^2 over all bins is the tapered variance."""
    x = np.nan_to_num(np.asarray(x, np.float64))
    w = np.asarray(weights, np.float64)
    a = (x - (x * w).sum((-2, -1), keepdims=True) / w.sum()) * w
    return np.fft.fft2(a) / np.sqrt(w.size * (w * w).sum())


def band_sums(forecast: np.ndarray, truth: np.ndarray, weights: np.ndarray, band: np.ndarray):
    """(sff, syy, sfy), each of the leading shape of forecast (..., ny, nx): the band's sums of the band-passed
    forecast f and truth y, f f, y y and f y. band is a boolean FFT-bin mask (band_mask)."""
    f, y = _spectrum(forecast, weights)[..., band], _spectrum(truth, weights)[..., band]
    return (np.abs(f) ** 2).sum(-1), (np.abs(y) ** 2).sum(-1), (f * y.conj()).real.sum(-1)


def slope_stats(sff, syy, sfy) -> dict:
    """slope, var_ratio and corr from band sums pooled by summation."""
    sff, syy, sfy = (float(np.sum(s)) for s in (sff, syy, sfy))
    return {"slope": sfy / sff, "var_ratio": sff / syy, "corr": sfy / np.sqrt(sff * syy)}


def band_slope(forecast: np.ndarray, truth: np.ndarray, weights: np.ndarray, band: np.ndarray) -> dict:
    """slope of the band-passed truth on the band-passed forecast, the variance ratio and the correlation, pooled
    over every leading dimension (starts) and every point."""
    return slope_stats(*band_sums(forecast, truth, weights, band))


def bootstrap_slope(sff, syy, sfy, n: int = 1000, q: tuple[float, float] = (0.05, 0.95), seed: int = 0):
    """(lo, hi) quantiles of the pooled slope over n resamples of the starts (the first axis of each sum)."""
    sff, sfy = np.asarray(sff, np.float64), np.asarray(sfy, np.float64)
    pick = np.random.default_rng(seed).integers(0, len(sff), (n, len(sff)))
    slopes = sfy[pick].sum(1) / sff[pick].sum(1)
    return tuple(float(v) for v in np.quantile(slopes, q))


def _band_power(x: np.ndarray, weights: np.ndarray, bins: np.ndarray) -> np.ndarray:
    return (np.abs(_spectrum(x, weights)) ** 2)[..., bins].sum(-1)


def alias_share(full: np.ndarray, stride: int, band: str, dx_km: float, mask: np.ndarray | None = None) -> np.ndarray:
    """Folded share (P_sub - P_full) / P_sub per leading index of full (..., ny, nx).

    P_sub: the power in the band of the field taken every stride-th point (as rea_pack reads it). P_full: the power
    of the full-resolution field over the same wavenumbers, the band's bins that lie inside the subsample's
    Nyquist square. mask: the points where the field is defined (default: where it is finite)."""
    full = np.asarray(full, np.float64)
    mask = np.isfinite(full).all(tuple(range(full.ndim - 2))) if mask is None else np.asarray(mask, bool)
    sub, sub_mask = full[..., ::stride, ::stride], mask[::stride, ::stride]
    p_sub = _band_power(sub, hann_taper(sub_mask), band_mask(band_index(sub.shape[-2:], dx_km * stride), band))
    nyquist = 1.0 / (2 * dx_km * stride)
    ky, kx = np.meshgrid(np.fft.fftfreq(full.shape[-2], dx_km), np.fft.fftfreq(full.shape[-1], dx_km), indexing="ij")
    inside = (np.abs(ky) <= nyquist) & (np.abs(kx) <= nyquist)
    bins = band_mask(band_index(full.shape[-2:], dx_km), band) & inside
    p_full = _band_power(full, hann_taper(mask), bins)
    return (p_sub - p_full) / p_sub


def ensemble_limit(mse_by_m: dict[int, float]) -> float:
    """Intercept at 1/M = 0 of the least-squares line of mean squared error against 1/M."""
    m = np.array(sorted(mse_by_m), np.float64)
    return float(np.polyfit(1.0 / m, [mse_by_m[k] for k in sorted(mse_by_m)], 1)[1])
