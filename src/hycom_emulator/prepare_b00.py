"""Turn a system store into the stacked layout neural-lam trains on, for module B00.

B00 steps the 00Z archv snapshot one day: state(00Z d) -> state(00Z d+1). Store row c holds the
snapshot at 00Z of day c and the increment valid at 18Z of day c. Over 00Z(d-1) -> 00Z(d) the IAU
adds 18 h of inc(d-2) then 6 h of inc(d-1), so the forcing at time 00Z(d) is
0.75 inc(d-2) + 0.25 inc(d-1). The first two rows have no forcing and are dropped.
With --atm (a forcing.py zarr of 6 h blocks), the forcing also holds the 24 h mean of each
atmospheric field over the same step, the four blocks that start at T-24h .. T-6h.

Output zarr: state (time, grid_index, state_feature), forcing (time, grid_index, forcing_feature),
static (grid_index, static_feature), boundary_mask (grid_index): 1 on land and in the nest band
(relax e-folding < BAND_EFOLD_DAYS). Land values are 0; neural-lam excludes boundary points from
the loss and overwrites them with these values. Statistics cover ocean points of the train split.
neural-lam weights each channel's loss by 1/(state_diff_std/state_std)^2. The near-fixed top
thickness layers barely change, so their diff std is floored at DIFF_STD_FLOOR x state_std, the
smallest ratio any T, S, u, v or SSH channel has; otherwise they swamp the loss.

Run `python -m hycom_emulator.prepare_b00 <store.zarr> <out.zarr> <relax.rmu.a> --train-end YYYY-MM-DD [--stride N]`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BAND_EFOLD_DAYS = 10.0
DIFF_STD_FLOOR = 0.05
LAYER_STATE = ("temp", "salin", "thknss", "u", "v")
SURFACE_STATE = ("srfhgt", "montg1", "ubaro", "vbaro")
LAYER_FORCING = ("temp", "salin", "thknss")
IAU_WEIGHTS = (0.75, 0.25)  # inc(d-2), inc(d-1) over 00Z(d-1) -> 00Z(d)
ATM_UNITS = {"wndewd": "m/s", "wndnwd": "m/s", "airtmp": "degC", "vapmix": "kg/kg", "precip": "m/s",
             "dswflx": "W/m2", "dlwflx": "W/m2", "mslprs": "Pa minus prsbas", "wndspd": "m/s",
             "wndspd_ewd": "m2/s2", "wndspd_nwd": "m2/s2"}
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
    from gom_da.eval.hycom_archive import Grid, read_rec  # RCC only; stack_b00 runs on skynet

    grid = Grid(nx, ny, 1, np.empty(0), np.empty(0))
    rmu = read_rec(rmu_a, 0, grid)
    rmu = np.where(np.abs(rmu) < 1e29, rmu, 0.0)
    return (rmu > 1.0 / (BAND_EFOLD_DAYS * 86400.0))[sl]


def prepare(store: Path, out: Path, rmu_a: Path, train_end: np.datetime64, stride: int = 1, atm: Path | None = None) -> None:
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
    atm_ds = xr.open_zarr(atm, consolidated=False) if atm is not None else None
    atm_names = sorted(atm_ds.data_vars) if atm_ds is not None else []
    f_names += atm_names
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
            "forcing_feature_units": (("forcing_feature",), [_forcing_unit(n) for n in f_names]),
            "static_feature_units": (("static_feature",), ["m", "degrees_east", "degrees_north", "1/s", "1"]),
        },
        attrs={**src.attrs, "stride": stride, "train_end": str(train_end), "iau_weights": str(IAU_WEIGHTS)},
    )
    # Without an explicit dtype this xarray writes the float32 template as float64.
    f32 = {"dtype": "float32"}
    template.to_zarr(out, mode="w", compute=False, consolidated=False, encoding={"state": f32, "forcing": f32})

    stats = TrainStats()
    incs = [_stack(src.isel(cycle=c), LAYER_FORCING, (), "inc_", sl) for c in (0, 1)]
    for n in range(t.size):
        row = src.isel(cycle=n + 2)
        st = np.nan_to_num(_to_grid_index(_stack(row, LAYER_STATE, SURFACE_STATE, "s00_", sl))).astype(np.float32)
        fo = _to_grid_index(IAU_WEIGHTS[0] * incs[0] + IAU_WEIGHTS[1] * incs[1])
        if atm_ds is not None:
            fo = np.concatenate([fo, _to_grid_index(_atm_24h(atm_ds, atm_names, t[n], sl))], axis=1)
        fo = np.nan_to_num(fo).astype(np.float32)
        incs = [incs[1], _stack(row, LAYER_FORCING, (), "inc_", sl)]
        xr.Dataset(
            {"state": (("time", g, "state_feature"), st[None]), "forcing": (("time", g, "forcing_feature"), fo[None])}
        ).to_zarr(out, region={"time": slice(n, n + 1)}, consolidated=False)
        if t[n] <= train_end:
            stats.add(st[ocean_gi], fo[ocean_gi])
        else:
            stats.new_run()
    stats.dataset(static[ocean_gi]).to_zarr(out, mode="a", consolidated=True)


def _forcing_unit(name: str) -> str:
    if name.startswith("atm_"):
        return ATM_UNITS[name[4:]]
    return UNITS[name[4:].split("_k")[0]]


def _atm_24h(atm_ds, names, t, sl) -> np.ndarray:
    """(feature, y, x) mean of the four 6 h blocks covering (t-24h, t]."""
    starts = [t - np.timedelta64(h, "h") for h in (24, 18, 12, 6)]
    blocks = atm_ds[names].sel(time=starts)
    return np.stack([blocks[n].values.mean(axis=0)[sl] for n in names])


def _safe(s):
    return np.where(s > 0, s, 1.0).astype(np.float32)


class TrainStats:
    """The statistics a B00 zarr stores, over the ocean points of train rows given in time order.
    Changes count only between consecutive rows of one run: call new_run() at a gap or a new run."""

    def __init__(self):
        self.acc = {k: _Moments() for k in ("state", "diff", "forcing")}
        self.prev = None

    def new_run(self):
        self.prev = None

    def add(self, state: np.ndarray, forcing: np.ndarray) -> None:
        """One row: state (points, state_feature) and forcing (points, forcing_feature)."""
        self.acc["state"].add(state)
        self.acc["forcing"].add(forcing)
        if self.prev is not None:
            self.acc["diff"].add(state - self.prev)
        self.prev = state

    def dataset(self, static: np.ndarray):
        """The statistics as a Dataset; static is (ocean points, static_feature)."""
        import xarray as xr

        a = self.acc
        # Floor the raw stds, before _safe's placeholder 1: a channel that never changes gets the floor,
        # one that is constant everywhere keeps 1 for both.
        raw = {k: np.sqrt(m.m2 / m.n).astype(np.float32) for k, m in a.items()}
        return xr.Dataset(
            {
                "state_mean": (("state_feature",), a["state"].mean),
                "state_std": (("state_feature",), a["state"].std),
                "state_diff_mean": (("state_feature",), a["diff"].mean),
                "state_diff_std": (("state_feature",), _safe(np.maximum(raw["diff"], DIFF_STD_FLOOR * raw["state"]))),
                "forcing_mean": (("forcing_feature",), a["forcing"].mean),
                "forcing_std": (("forcing_feature",), a["forcing"].std),
                "static_mean": (("static_feature",), static.mean(axis=0)),
                "static_std": (("static_feature",), _safe(static.std(axis=0))),
            }
        )


class _Moments:
    """Running per-feature mean and std over rows of (points, feature) arrays.
    Merges per-batch mean and squared deviations (Chan et al.), so a constant feature gets std 0
    instead of the cancellation residue of E[x^2] - m^2."""

    def __init__(self):
        self.n, self.m, self.m2 = 0, 0.0, 0.0

    def add(self, a):
        a = a.astype(np.float64)
        nb = a.shape[0]
        mb = a.mean(axis=0)
        n = self.n + nb
        delta = mb - self.m
        self.m2 = self.m2 + ((a - mb) ** 2).sum(axis=0) + delta**2 * self.n * nb / n
        self.m = self.m + delta * nb / n
        self.n = n

    @property
    def mean(self):
        return np.asarray(self.m, dtype=np.float32)

    @property
    def std(self):
        return _safe(np.sqrt(self.m2 / self.n))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("store", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("rmu", type=Path)
    p.add_argument("--train-end", required=True, type=np.datetime64)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--atm", type=Path, default=None, help="forcing.py zarr of 6 h atmospheric blocks")
    a = p.parse_args()
    prepare(a.store, a.out, a.rmu, a.train_end, a.stride, a.atm)


if __name__ == "__main__":
    main()
