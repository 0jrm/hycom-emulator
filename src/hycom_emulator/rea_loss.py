"""`rea_wmse`: neural-lam's wmse plus a penalty on the Gulf-mean error of each conserved quantity (term A of docs/conservation.md).

    loss = wmse + mean_penalty * sum_q (Q_q(e) / s_q)^2 + amse * amse_excess + stability * inversion_penalty

per (batch, step), where Q_q is a `conservation.functionals` quantity over the Gulf (ssh, ubaro and vbaro area means;
temp, salin, u and v volume means to 2000 m) of the physical error e, and s_q the train std of the quantity's one-day
change (`scales_from_series`). inversion_penalty (Stability) needs no target: the truth's sigma2 falls downward on
about 0.01% of adjacent level pairs, the free runs' on 1-5% (gom-da runs/emu-rea-phys0). amse_excess turns wmse into the adjusted MSE of Subich et al. (2025), which stops
paying the model for smoothing what it cannot place. Each added term is skipped when its weight is 0, so both
weights 0 is wmse exactly.

Importing this module registers `rea_wmse` with neural-lam. hycom_emulator.rea_train configures it from its CLI flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from neural_lam import metrics

from hycom_emulator.conservation import functionals
from hycom_emulator.evaluate_rea import LEVEL_VARS, cell_area, regions
from hycom_emulator.physics import sigma2

EPS = 1e-20  # keeps sqrt(Px * Py) differentiable where a band has no power; computed in float64
INVERSION_SCALE = 0.01  # kg/m3: a downward sigma2 drop of this size on one level pair costs 1
QUANTITIES = ("ssh_mean", "ubaro_mean", "vbaro_mean", "temp_mean", "salin_mean", "u_mean", "v_mean")


@dataclass(frozen=True)
class DomainMeans:
    """Gulf-mean errors of QUANTITIES from standardized states: errors(pred, target) = ((pred - target) * weight).sum(-2) @ group."""

    names: tuple[str, ...]
    weight: torch.Tensor  # (grid, feature): functional weight times state_std, 0 outside the Gulf and below the floor
    group: torch.Tensor  # (feature, quantity) one-hot

    @classmethod
    def from_meta(cls, meta: xr.Dataset) -> DomainMeans:
        names = [str(n) for n in meta.state_feature.values]
        fs = functionals(names, cell_area(meta), regions(meta)["gulf"], meta.level_ocean.values.astype(bool), meta.level.values.astype(float))
        by_name = {f.name: f for f in fs}
        std = meta.state_std.values.astype(float)
        weight = np.zeros((meta.sizes["grid_index"], len(names)))
        group = np.zeros((len(names), len(QUANTITIES)))
        for q, name in enumerate(QUANTITIES):
            f = by_name[name]
            weight[:, f.cols] = f.w * std[f.cols]
            group[f.cols, q] = 1.0
        return cls(QUANTITIES, torch.tensor(weight, dtype=torch.float32), torch.tensor(group, dtype=torch.float32))

    def to(self, device) -> DomainMeans:
        return DomainMeans(self.names, self.weight.to(device), self.group.to(device))

    def errors(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """(..., grid, feature) standardized -> (..., quantity) in SI units."""
        return ((pred - target) * self.weight).sum(-2) @ self.group


@dataclass(frozen=True)
class Stability:
    """Static stability of standardized states: penalty(pred) = mean over real level pairs in the interior of
    (relu(sigma2_k - sigma2_k+1) / INVERSION_SCALE)^2, levels top down, in float32 whatever the autocast precision."""

    temp: torch.Tensor  # (K,) column of temp at each level, top down
    salin: torch.Tensor  # (K,)
    mean: torch.Tensor  # (2K,) state_mean of temp then salin columns
    std: torch.Tensor  # (2K,)
    pairs: torch.Tensor  # (grid, K - 1) float32: 1 where both levels are real at an interior point

    @classmethod
    def from_meta(cls, meta: xr.Dataset) -> Stability:
        names = [str(n) for n in meta.state_feature.values]
        levels = meta.level.values.astype(float)
        col = {v: [names.index(f"{v}_{z:g}m") for z in levels] for v in LEVEL_VARS[:2]}
        cols = col["temp"] + col["salin"]
        real = meta.level_ocean.values.astype(bool) & regions(meta)["interior"][:, None]
        t = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt)  # noqa: E731
        return cls(t(col["temp"], torch.long), t(col["salin"], torch.long), t(meta.state_mean.values[cols]),
                   t(meta.state_std.values[cols]), t(real[:, 1:] & real[:, :-1]))

    def to(self, device) -> Stability:
        return Stability(*(getattr(self, f).to(device) for f in ("temp", "salin", "mean", "std", "pairs")))

    def penalty(self, pred: torch.Tensor) -> torch.Tensor:
        """(..., grid, feature) standardized -> (...,)."""
        x = pred[..., torch.cat([self.temp, self.salin])].float() * self.std + self.mean
        k = len(self.temp)
        rho = sigma2(x[..., :k], x[..., k:])
        drop = torch.relu(rho[..., :-1] - rho[..., 1:]) / INVERSION_SCALE
        return (drop**2 * self.pairs).sum((-2, -1)) / self.pairs.sum()


def scales_from_series(npz_path: Path, train_end) -> np.ndarray:
    """(quantity,) train std of the Gulf one-day change of QUANTITIES, from a `conservation series` npz: pairs of
    real rows whose later row is on or before train_end (whole days)."""
    z = np.load(npz_path)
    days = z["time"].astype("datetime64[D]")
    filled = z["time_filled"].astype(bool)
    ok = ~(filled[1:] | filled[:-1]) & (days[1:] <= np.datetime64(train_end, "D"))
    quantities = [str(q) for q in z["quantities"]]
    values = z["values"][:, [str(r) for r in z["regions"]].index("gulf")][:, [quantities.index(q) for q in QUANTITIES]]
    return np.diff(values, axis=0)[ok].std(0)


@lru_cache
def _bands(nx: int, ny: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """For an rfft2 of an (nx, ny) field, flattened: the isotropic band of each coefficient, round(|k| max(nx, ny))
    with k in cycles per cell, and its weight, 2 for the columns that stand for a conjugate pair as well. Then
    sum over bands of the weighted |X|^2 is sum(x^2) (Parseval, norm="ortho"). Returns (band, weight, bands)."""
    k = np.sqrt(np.fft.fftfreq(nx)[:, None] ** 2 + np.fft.rfftfreq(ny)[None] ** 2)
    band = np.rint(k * max(nx, ny)).astype(np.int64)
    weight = np.full(band.shape, 2.0)
    weight[:, 0] = 1.0
    if ny % 2 == 0:
        weight[:, -1] = 1.0
    return torch.tensor(band.ravel(), device=device), torch.tensor(weight.ravel(), dtype=torch.float32, device=device), int(band.max()) + 1


def amse_excess(pred, target, pred_std, mask, grid_shape: tuple[int, int]) -> torch.Tensor:
    """Adjusted MSE minus wmse (Subich et al. 2025), summed over channels: (..., grid, feature) -> (...,).

    Per channel, x = mask * pred / pred_std and y = mask * target / pred_std on the (nx, ny) grid have band powers Px,
    Py and cross power C over isotropic wavenumber bands. sum(x - y)^2 = sum_l (sqrt Px - sqrt Py)^2 + 2 g (1 - C/g)
    with g = sqrt(Px Py); the adjusted MSE takes max(Px, Py) for g in the decorrelation term, so a band that has lost
    coherence still costs its full power and smoothing it away no longer pays. The excess
    sum_l 2 (max(Px, Py) - g)(1 - C/g) is divided by the mask's point count, as wmse averages over it."""
    nx, ny = grid_shape
    m = torch.ones(nx * ny, dtype=pred.dtype, device=pred.device) if mask is None else mask.to(pred.dtype)
    band, weight, n_bands = _bands(nx, ny, pred.device)

    def spectrum(field):
        x = (field / pred_std * m[:, None]).movedim(-1, -2)  # (..., feature, grid)
        return torch.fft.rfft2(x.reshape(*x.shape[:-1], nx, ny), norm="ortho").flatten(-2)

    def binned(v):
        return torch.zeros(*v.shape[:-1], n_bands, dtype=v.dtype, device=v.device).index_add(-1, band, v * weight).double()

    X, Y = spectrum(pred), spectrum(target)
    px, py = binned(X.real**2 + X.imag**2), binned(Y.real**2 + Y.imag**2)
    c = binned(X.real * Y.real + X.imag * Y.imag)
    g = torch.sqrt(px * py + EPS)
    excess = (2 * (torch.maximum(px, py) - g) * (1 - c / g)).sum((-2, -1))
    return (excess / m.sum()).to(pred.dtype)


@dataclass
class _Settings:
    means: DomainMeans | None
    inv_scale: torch.Tensor | None  # (quantity,)
    mean_penalty: float
    amse: float
    grid_shape: tuple[int, int]
    stability: Stability | None = None
    stability_weight: float = 0.0

    def to(self, device) -> _Settings:
        if self.means is not None and self.inv_scale.device != device:
            self.means, self.inv_scale = self.means.to(device), self.inv_scale.to(device)
        if self.stability is not None and self.stability.pairs.device != device:
            self.stability = self.stability.to(device)
        return self


_SETTINGS: _Settings | None = None


def configure(means: DomainMeans | None, inv_scale: np.ndarray | None, mean_penalty: float, amse: float = 0.0,
              grid_shape: tuple[int, int] | None = None, stability: Stability | None = None,
              stability_weight: float = 0.0) -> None:
    global _SETTINGS
    if mean_penalty and (means is None or inv_scale is None):
        raise ValueError("a mean penalty needs the domain means and their scales")
    if amse and grid_shape is None:
        raise ValueError("the amse term needs the grid shape")
    if stability_weight and stability is None:
        raise ValueError("the stability term needs the level columns (Stability.from_meta)")
    inv = None if inv_scale is None else torch.tensor(np.asarray(inv_scale), dtype=torch.float32)
    _SETTINGS = _Settings(means, inv, float(mean_penalty), float(amse), grid_shape, stability, float(stability_weight))


def rea_wmse(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """wmse, plus the configured terms when the result is one loss per (batch, step); per-variable or per-point
    calls (logging, test maps) return wmse unchanged."""
    if _SETTINGS is None:
        raise RuntimeError("rea_wmse is not configured: train through `python -m hycom_emulator.nlam train_model --loss rea_wmse --mean_penalty ...`")
    loss = metrics.wmse(pred, target, pred_std, mask=mask, average_grid=average_grid, sum_vars=sum_vars)
    if not (sum_vars and average_grid):
        return loss
    s = _SETTINGS.to(pred.device)
    if s.mean_penalty:
        loss = loss + s.mean_penalty * ((s.means.errors(pred, target) * s.inv_scale) ** 2).sum(-1)
    if s.amse:
        loss = loss + s.amse * amse_excess(pred, target, pred_std, mask, s.grid_shape)
    if s.stability_weight:
        loss = loss + (s.stability_weight * s.stability.penalty(pred)).to(loss.dtype)
    return loss


metrics.DEFINED_METRICS["rea_wmse"] = rea_wmse
