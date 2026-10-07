"""`rea_wmse`: neural-lam's wmse plus a penalty on the Gulf-mean error of each conserved quantity (term A of docs/conservation.md).

    loss = wmse + mean_penalty * sum_q (Q_q(e) / s_q)^2

per (batch, step), where Q_q is a `conservation.functionals` quantity over the Gulf (ssh, ubaro and vbaro area means;
temp, salin, u and v volume means to 2000 m) of the physical error e, and s_q the train std of the quantity's one-day
change (`scales_from_series`). The added term is skipped when its weight is 0, so mean_penalty 0 is wmse exactly.

Importing this module registers `rea_wmse` with neural-lam. hycom_emulator.rea_train configures it from its CLI flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from neural_lam import metrics

from hycom_emulator.conservation import functionals
from hycom_emulator.evaluate_rea import cell_area, regions

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


@dataclass
class _Settings:
    means: DomainMeans | None
    inv_scale: torch.Tensor | None  # (quantity,)
    mean_penalty: float

    def to(self, device) -> _Settings:
        if self.means is not None and self.inv_scale.device != device:
            self.means, self.inv_scale = self.means.to(device), self.inv_scale.to(device)
        return self


_SETTINGS: _Settings | None = None


def configure(means: DomainMeans | None, inv_scale: np.ndarray | None, mean_penalty: float) -> None:
    global _SETTINGS
    if mean_penalty and (means is None or inv_scale is None):
        raise ValueError("a mean penalty needs the domain means and their scales")
    inv = None if inv_scale is None else torch.tensor(np.asarray(inv_scale), dtype=torch.float32)
    _SETTINGS = _Settings(means, inv, float(mean_penalty))


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
    return loss


metrics.DEFINED_METRICS["rea_wmse"] = rea_wmse
