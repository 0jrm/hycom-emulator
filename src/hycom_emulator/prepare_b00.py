"""Turn a system store into the stacked layout neural-lam trains on, for module B00.

B00 steps the 00Z archv snapshot one day: state(00Z d) -> state(00Z d+1). Store row c holds the
snapshot at 00Z of day c and the increment valid at 18Z of day c. Over 00Z(d-1) -> 00Z(d) the IAU
adds 18 h of inc(d-2) then 6 h of inc(d-1), so the forcing at time 00Z(d) is
0.75 inc(d-2) + 0.25 inc(d-1). The first two rows have no forcing and are dropped.

Output zarr: state (time, grid_index, state_feature), forcing (time, grid_index, forcing_feature),
static (grid_index, static_feature), boundary_mask (grid_index): 1 on land and in the nest band
(relax e-folding < BAND_EFOLD_DAYS). Land values are 0; neural-lam excludes boundary points from
the loss and overwrites them with these values. Statistics cover ocean points of the train split.

Run `python -m hycom_emulator.prepare_b00 <store.zarr> <out.zarr> <relax.rmu.a> --train-end YYYY-MM-DD [--stride N]`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gom_da.eval.hycom_archive import Grid, read_rec

BAND_EFOLD_DAYS = 10.0
LAYER_STATE = ("temp", "salin", "thknss", "u", "v")
SURFACE_STATE = ("srfhgt", "montg1", "ubaro", "vbaro")
LAYER_FORCING = ("temp", "salin", "thknss")
IAU_WEIGHTS = (0.75, 0.25)  # inc(d-2), inc(d-1) over 00Z(d-1) -> 00Z(d)
UNITS = {"temp": "degC", "salin": "psu", "thknss": "Pa", "u": "m/s", "v": "m/s", "srfhgt": "m*g", "montg1": "m*g", "ubaro": "m/s", "vbaro": "m/s"}


def feature_names(layer_vars, surface_vars, nlayer, prefix=""):
    names = [f"{prefix}{v}_k{k + 1:02d}" for v in layer_vars for k in range(nlayer)]
    return names + [f"{prefix}{v}" for v in surface_vars]


def _stack(ds_row, layer_vars, surface_vars, prefix, sl):
    """(feature, y, x) for one store row, layers first then surface fields."""
    parts = [ds_row[f"{prefix}{v}"].values[:, sl[0], sl[1]] for v in layer_vars]
    parts += [ds_row[f"{prefix}{v}"].values[None, sl[0], sl[1]] for v in surface_vars]
    return np.concatenate(parts, axis=0)


def _to_grid_index(a):
    """(feature, y, x) -> (x*y, feature), x outer, as neural-lam stacks ("x", "y")."""
    return a.transpose(2, 1, 0).reshape(-1, a.shape[0])


def band_mask(rmu_a: Path, ny: int, nx: int, sl) -> np.ndarray:
    grid = Grid(nx, ny, 1, np.empty(0), np.empty(0))
    rmu = read_rec(rmu_a, 0, grid)
    rmu = np.where(np.abs(rmu) < 1e29, rmu, 0.0)
    return (rmu > 1.0 / (BAND_EFOLD_DAYS * 86400.0))[sl]


def prepare(store: Path, out: Path, rmu_a: Path, train_end: np.datetime64, stride: int = 1) -> None:
    import dask.array as da
    import xarray as xr

    src = xr.open_zarr(store, consolidated=False)
    ny, nx = src.sizes["y"], src.sizes["x"]
    nlayer = src.sizes["layer"]
    sl = (slice(None, None, stride), slice(None, None, stride))
    ocean = src["ocean"].values[sl]
    boundary = ~ocean | band_mask(rmu_a, ny, nx, sl)
    lon, lat = src["plon"].values[sl], src["plat"].values[sl]
    x, y = lon[0, :], lat[:, 0]

    s_names = feature_names(LAYER_STATE, SURFACE_STATE, nlayer)
    f_names = feature_names(LAYER_FORCING, (), nlayer, prefix="inc_")
    t = (src["cycle"].values - np.timedelta64(18, "h"))[2:]  # snapshot time of each kept row
    ngrid = x.size * y.size
    gx, gy = np.meshgrid(x, y, indexing="ij")
    ocean_gi = ocean.T.reshape(-1)
    static_vars = {
        "depth": np.nan_to_num(src["depth"].values[sl]),
        "lon": lon,
        "lat": lat,
        "coriolis": 2 * 7.2921e-5 * np.sin(np.deg2rad(lat)),
        "ocean": ocean.astype(np.float32),
    }
    static = np.stack([v.T.reshape(-1) for v in static_vars.values()], axis=1).astype(np.float32)

    g = "grid_index"
    template = xr.Dataset(
        {
            "state": (("time", g, "state_feature"), da.zeros((t.size, ngrid, len(s_names)), np.float32, chunks=(1, -1, -1))),
            "forcing": (("time", g, "forcing_feature"), da.zeros((t.size, ngrid, len(f_names)), np.float32, chunks=(1, -1, -1))),
            "static": ((g, "static_feature"), static),
            "boundary_mask": ((g,), boundary.T.reshape(-1).astype(np.int8)),
        },
        coords={
            "time": t,
            "x": ((g,), gx.reshape(-1)),
            "y": ((g,), gy.reshape(-1)),
            "state_feature": s_names,
            "forcing_feature": f_names,
            "static_feature": list(static_vars),
            "state_feature_units": (("state_feature",), [UNITS[n.split("_k")[0]] for n in s_names]),
            "forcing_feature_units": (("forcing_feature",), [UNITS[n[4:].split("_k")[0]] for n in f_names]),
            "static_feature_units": (("static_feature",), ["m", "degrees_east", "degrees_north", "1/s", "1"]),
        },
        attrs={**src.attrs, "stride": stride, "train_end": str(train_end), "iau_weights": str(IAU_WEIGHTS)},
    )
    template.to_zarr(out, mode="w", compute=False, consolidated=False)

    acc = {k: _Moments() for k in ("state", "diff", "forcing")}
    incs = [_stack(src.isel(cycle=c), LAYER_FORCING, (), "inc_", sl) for c in (0, 1)]
    prev = None
    for n in range(t.size):
        row = src.isel(cycle=n + 2)
        st = np.nan_to_num(_to_grid_index(_stack(row, LAYER_STATE, SURFACE_STATE, "s00_", sl)))
        fo = np.nan_to_num(_to_grid_index(IAU_WEIGHTS[0] * incs[0] + IAU_WEIGHTS[1] * incs[1]))
        incs = [incs[1], _stack(row, LAYER_FORCING, (), "inc_", sl)]
        xr.Dataset(
            {"state": (("time", g, "state_feature"), st[None]), "forcing": (("time", g, "forcing_feature"), fo[None])}
        ).to_zarr(out, region={"time": slice(n, n + 1)}, consolidated=False)
        if t[n] <= train_end:
            acc["state"].add(st[ocean_gi])
            acc["forcing"].add(fo[ocean_gi])
            if prev is not None:
                acc["diff"].add(st[ocean_gi] - prev)
            prev = st[ocean_gi]
        else:
            prev = None

    s_ocean = static[ocean_gi]
    stats = xr.Dataset(
        {
            "state_mean": (("state_feature",), acc["state"].mean),
            "state_std": (("state_feature",), acc["state"].std),
            "state_diff_mean": (("state_feature",), acc["diff"].mean),
            "state_diff_std": (("state_feature",), acc["diff"].std),
            "forcing_mean": (("forcing_feature",), acc["forcing"].mean),
            "forcing_std": (("forcing_feature",), acc["forcing"].std),
            "static_mean": (("static_feature",), s_ocean.mean(axis=0)),
            "static_std": (("static_feature",), _safe(s_ocean.std(axis=0))),
        }
    )
    stats.to_zarr(out, mode="a", consolidated=True)


def _safe(s):
    return np.where(s > 0, s, 1.0).astype(np.float32)


class _Moments:
    """Running per-feature mean and std over rows of (points, feature) arrays."""

    def __init__(self):
        self.n, self.s, self.ss = 0, 0.0, 0.0

    def add(self, a):
        a = a.astype(np.float64)
        self.n += a.shape[0]
        self.s = self.s + a.sum(axis=0)
        self.ss = self.ss + (a * a).sum(axis=0)

    @property
    def mean(self):
        return (self.s / self.n).astype(np.float32)

    @property
    def std(self):
        m = self.s / self.n
        return _safe(np.sqrt(np.maximum(self.ss / self.n - m * m, 0.0)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("store", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("rmu", type=Path)
    p.add_argument("--train-end", required=True, type=np.datetime64)
    p.add_argument("--stride", type=int, default=1)
    a = p.parse_args()
    prepare(a.store, a.out, a.rmu, a.train_end, a.stride)


if __name__ == "__main__":
    main()
