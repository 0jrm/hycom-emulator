"""Write one system's cycles into a zarr store the training code reads.

Root dataset, dims (cycle, layer, y, x), land is NaN, float32:
  static   plon plat pscx pscy depth ocean
  xb_*     24 h-mean TSIS background (background.py): temp salin thknss srfhgt montg1 oneta
  inc_*    TSIS increment from inc.*.nc: temp salin thknss ssh (thknss in Pa; the file says m)
  s00_*    archv snapshot at t_a-18h (00Z): temp salin thknss u v srfhgt montg1 ubaro vbaro oneta
  l3sst l3sst_err l3ssh l3ssh_err   gridded obs from tsis_obs
Group obs/<YYYYMMDDHH of t_a>: one row per valid layer obs (tsis.LayerObs fields).
A free run (assimilation = false) stores only static, s00_* and inc_* = 0 over ocean, so its rows
line up with a cycled run's for prepare_b00.

Rerunning skips cycles already in the store, so an interrupted build resumes where it stopped.
Run `python -m hycom_emulator.build_store <system.toml> <out.zarr> [first last]` (dates YYYY-MM-DD).
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

from gom_da.eval.hycom_archive import Grid, load_2d, load_3d, parse_archv_index, read_rec

from hycom_emulator.background import cycle_background, grid_of
from hycom_emulator.catalog import Cycle, cycles
from hycom_emulator.system import SystemConfig, fingerprint
from hycom_emulator.tsis import FILL_ABS, read_layer_obs

LAND = 1e29  # HYCOM spval is 2**100
ROLES_NEEDED = ("archm_21", "archm_09", "archv_00", "inc_nc", "tsis_obs")
ROLES_NEEDED_FREE = ("archv_00",)
S00_LAYER = {"temp": "temp", "salin": "salin", "thknss": "thknss", "u": "u-vel.", "v": "v-vel."}
S00_SURFACE = {"srfhgt": "srfhgt", "montg1": "montg1", "ubaro": "u_btrop", "vbaro": "v_btrop", "oneta": "oneta"}
INC = {"temp": "tem", "salin": "sal", "thknss": "thk", "ssh": "ssh"}
GRIDDED_OBS = ("l3sst", "l3sst_err", "l3ssh", "l3ssh_err")
L3 = ("y", "x")
L4 = ("layer", "y", "x")


def _f32(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return np.where(np.abs(a) < LAND, a, np.nan).astype(np.float32)


def _grid_records(apath: Path, grid: Grid) -> dict[str, np.ndarray]:
    """regional.grid / regional.depth: `name: min,max` lines in .b, one record each in .a."""
    names = [ln.split(":")[0].strip() for ln in apath.with_suffix(".b").read_text().splitlines() if ":" in ln]
    return {n: read_rec(apath, i, grid) for i, n in enumerate(names)}


def statics(cfg: SystemConfig, grid: Grid):
    import xarray as xr

    data = cfg.hycom_exe.parent
    g = _grid_records(data / "regional.grid.a", grid)
    depth = _f32(read_rec(data / "regional.depth.a", 0, grid))
    return xr.Dataset(
        {
            **{n: (L3, _f32(g[n])) for n in ("plon", "plat", "pscx", "pscy")},
            "depth": (L3, depth),
            "ocean": (L3, np.isfinite(depth) & (depth > 0)),
        }
    )


def cycle_dataset(cycle: Cycle, grid: Grid, assimilation: bool = True):
    import netCDF4
    import xarray as xr

    v: dict[str, tuple] = {}
    a = cycle.files["archv_00"]
    idx = parse_archv_index(a.with_suffix(".b"))
    v.update({f"s00_{n}": (L4, _f32(load_3d(a, idx, f, grid))) for n, f in S00_LAYER.items()})
    v.update({f"s00_{n}": (L3, _f32(load_2d(a, idx, f, grid))) for n, f in S00_SURFACE.items()})
    if not assimilation:
        wet = np.where(np.isfinite(v["s00_temp"][1]), 0.0, np.nan).astype(np.float32)
        v.update({f"inc_{n}": (L4, wet) for n in ("temp", "salin", "thknss")})
        v["inc_ssh"] = (L3, wet[0])
        out = xr.Dataset({n: (("cycle", *dims), x[None]) for n, (dims, x) in v.items()})
        return out.assign_coords(cycle=[np.datetime64(cycle.analysis, "ns")])
    xb = cycle_background(cycle, grid)
    v.update({f"xb_{n}": (L4, _f32(x)) for n, x in xb.layer.items()})
    v["xb_thknss"] = (L4, _f32(xb.thknss))
    v["xb_oneta"] = (L3, _f32(xb.oneta))
    v.update({f"xb_{n}": (L3, _f32(x)) for n, x in xb.surface.items()})
    with netCDF4.Dataset(cycle.files["inc_nc"]) as ds:
        ds.set_auto_mask(False)
        for n, f in INC.items():
            x = np.asarray(ds[f][:], dtype=np.float64)
            v[f"inc_{n}"] = (L4 if x.ndim == 3 else L3, np.where(np.abs(x) < FILL_ABS, x, np.nan).astype(np.float32))
    with netCDF4.Dataset(cycle.files["tsis_obs"]) as ds:
        ds.set_auto_mask(False)
        for n in GRIDDED_OBS:
            x = np.asarray(ds[n][:], dtype=np.float64)
            v[n] = (L3, np.where(np.abs(x) < FILL_ABS, x, np.nan).astype(np.float32))
    out = xr.Dataset({n: (("cycle", *dims), x[None]) for n, (dims, x) in v.items()})
    return out.assign_coords(cycle=[np.datetime64(cycle.analysis, "ns")])


def obs_dataset(cycle: Cycle):
    import xarray as xr

    o = read_layer_obs(cycle.files["tsis_obs"])
    return xr.Dataset({f: ("row", getattr(o, f)) for f in o.__dataclass_fields__})


def written_cycles(out: Path, attrs: dict[str, str]) -> set[np.datetime64]:
    """Cycles already in the store. Refuses a store written for another system configuration."""
    import xarray as xr

    if not (out / "zarr.json").exists():
        return set()
    with xr.open_zarr(out, consolidated=False) as ds:
        if ds.attrs.get("structural") != attrs["structural"]:
            raise RuntimeError(f"{out} holds system {ds.attrs.get('system')}, not {attrs['system']}")
        lengths = {ds[n].sizes["cycle"] for n in ds.data_vars if "cycle" in ds[n].dims}
        if len(lengths) > 1:
            raise RuntimeError(f"{out}: variables disagree on cycle length {lengths}; an append was cut short")
        return set(ds["cycle"].values)


def build(cfg: SystemConfig, out: Path, first: date | None = None, last: date | None = None) -> list[str]:
    grid = grid_of(cfg)
    fp = fingerprint(cfg)
    # Every write carries the attrs: xarray replaces the root attrs on append.
    attrs = dict(system=cfg.name, canonical=str(cfg.canonical), structural=fp.structural, strict=fp.strict)
    done = written_cycles(out, attrs)
    log = []
    for cycle in cycles(cfg):
        day = cycle.analysis.date()
        if (first and day < first) or (last and day > last):
            continue
        stamp = f"{cycle.analysis:%Y%m%d%H}"
        if np.datetime64(cycle.analysis, "ns") in done:
            log.append(f"{stamp} already written")
            continue
        missing = cycle.missing(ROLES_NEEDED if cfg.assimilation else ROLES_NEEDED_FREE)
        if missing:
            log.append(f"{stamp} SKIP missing {missing}")
            continue
        t0 = time.time()
        ds = cycle_dataset(cycle, grid, cfg.assimilation)
        if not done:
            # mode="w" clears the whole store, so the root goes first and obs groups after it.
            ds = ds.merge(statics(cfg, grid))
            ds.attrs.update(attrs)
            ds.to_zarr(out, mode="w", consolidated=False)
        else:
            ds.attrs.update(attrs)
            ds.to_zarr(out, append_dim="cycle", consolidated=False)
        if cfg.assimilation:
            obs_dataset(cycle).to_zarr(out, group=f"obs/{stamp}", mode="w", consolidated=False)
        done.add(ds["cycle"].values[0])
        log.append(f"{stamp} wrote in {time.time() - t0:.1f}s")
        print(log[-1], flush=True)
    return log


def main(argv: list[str]) -> None:
    cfg = SystemConfig.from_toml(Path(argv[1]))
    first, last = (date.fromisoformat(a) for a in argv[3:5]) if len(argv) >= 5 else (None, None)
    for line in build(cfg, Path(argv[2]), first, last):
        if "wrote" not in line:
            print(line)


if __name__ == "__main__":
    main(sys.argv)
