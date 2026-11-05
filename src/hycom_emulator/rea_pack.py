"""Pack the public GOMb0.04 reanalysis daily means into the folder training reads from RAM.

    out/state.npy    (time, grid_index, state_feature)   float32 (int16 with --dtype int16)
    out/forcing.npy  (time, grid_index, forcing_feature) float32 (int16 with --dtype int16)
    out/meta.zarr    coordinates, static fields, boundary mask and train statistics, as pack_b00
                     writes them, plus level_ocean (grid_index, level) and time_filled (time)

The source is one pair of netCDF-4 files per day under <root>/YYYY/: gomb4_daily_YYYY_DDD_3z.nc
(u, v, w_velocity, water_temp, salinity on 40 z levels) and gomb4_daily_YYYY_DDD_2d.nc (ssh,
barotropic velocity, mixed layer, 10 m wind). Row n holds day n's daily mean, stamped at its 12 h centre.
State is T, S, u, v on the selected z levels, then ssh and barotropic velocity; forcing is that
day's mean wind and three calendar channels. Below the bottom and on land each state channel holds one
constant, its mean over real points on the start day, so those points standardize near 0 and never
change; level_ocean marks where each level is real. The statistics use real points
of real train rows only. A day whose files are missing is interpolated in time and flagged in
time_filled; a gap longer than --max-gap refuses the plan.

The build resumes. out/plan.json fixes the plan, out/written.npy marks finished rows, and meta.zarr
is written last, so meta.zarr present means the pack is complete. Rerun the same command after a crash.

--dtype int16 then stores state and forcing as standardised fixed point, q = round((x - mean) / std * INT16_SCALE),
clipped at +-INT16_CLIP std (resolution 1/INT16_SCALE std, 1% of the smallest one-day change std at the 0.05 floor),
which halves the pack and the loader's shared memory. Statistics come from the float32 rows first; each array is then
converted in place and meta.zarr's attrs record the scale, the clip and the clipped counts last, so the attrs mark a
finished conversion. The datastore converts back per sample. A float32 pack in the folder converts on a rerun with
--dtype int16.

Each level variable's state std is floored (STATE_STD_MIN, STATE_STD_MEDIAN_FRACTION): one Gulf-wide std per channel
is tiny at depth, so ordinary bottom-layer variability standardized with it became 35 sigma inputs, and a free run of
rcan_s2 fed one back until it blew up (emu-rea-rcan incident 2; docs/redesign-2026-11-04-rcan-first-principles.md
2.1). meta.zarr keeps the unfloored std as state_std_raw and lists the floored channels and the floor per variable in
its attrs. The change std keeps its floor at DIFF_STD_FLOOR x the raw state std, so state_diff_std is unchanged and so
is every channel's loss weight in physical units (neural-lam's wmse weights a standardized error by
state_std / state_diff_std, which is 1 / state_diff_std on the physical error). `refloor` writes the floored statistics
for a pack built before the floor into a new folder that links the old arrays; for an int16 pack the std the values
were quantized with stays in meta.zarr as state_int16_std and the datastore decodes with it. `zscan` reports the
largest standardized |state| per channel over a sample of the train rows, before and after the floor.

Run `python -m hycom_emulator.rea_pack build <out_dir> --start YYYY-MM-DD --end YYYY-MM-DD --train-end YYYY-MM-DD`.
`python -m hycom_emulator.rea_pack inventory` lists which days the archive holds.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import re
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage

from hycom_emulator.prepare_b00 import DIFF_STD_FLOOR, _Moments, _safe, _to_grid_index

DEFAULT_ROOT = "/hycom/ftp/pub/BOEM/GOMb0.04/data/daily_netcdf"
DEFAULT_TOPO = "/hycom/ftp/pub/BOEM/GOMb0.04/topo/regional.depth.a"
FORMAT = 1
INT16_SCALE = 2000  # counts per standard deviation
INT16_CLIP = 16  # standard deviations
FILL = 1e29
# State std floor per level variable: max(physical minimum, STATE_STD_MEDIAN_FRACTION x the median std over the column).
STATE_STD_MIN = {"temp": 0.05, "salin": 0.02, "u": 0.01, "v": 0.01}  # degC, psu, m/s
STATE_STD_MEDIAN_FRACTION = 0.1
# A level that sits on the sea floor can be fill on some days and valid on others (6 points at 800 m on the
# Blake Plateau in 2001); such points take the value of the level above. More than this fraction is a bad file.
MAX_LOST_FRACTION = 1e-3
DEPTH_AXIS = (0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 125, 150, 200, 250, 300,
              350, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000)
DEFAULT_DEPTHS = (0, 4, 10, 20, 30, 40, 50, 70, 100, 125, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1250, 1500, 2000)

LEVEL_VARS = {"temp": "water_temp", "salin": "salinity", "u": "u", "v": "v"}
SURFACE_VARS = {"ssh": "ssh", "ubaro": "u_barotropic_velocity", "vbaro": "v_barotropic_velocity"}
WIND_VARS = {"wnd_ewd": "wnd_ewd", "wnd_nwd": "wnd_nwd"}
CALENDAR = ("sin_doy", "cos_doy", "insolation")
STATIC_UNITS = {"depth": "m", "lon": "degrees_east", "lat": "degrees_north", "coriolis": "1/s", "ocean": "1", "gulf": "1"}
UNITS = {"temp": "degC", "salin": "psu", "u": "m/s", "v": "m/s", "ssh": "m", "ubaro": "m/s", "vbaro": "m/s",
         "wnd_ewd": "m/s", "wnd_nwd": "m/s", "sin_doy": "1", "cos_doy": "1", "insolation": "W/m2"}

GULF_SEED = (-90.0, 25.0)
GULF_SECTIONS = (((-87.1, 21.5), (-84.8, 21.95)), ((-81.1, 25.4), (-81.1, 22.9)))  # Yucatan Channel, Florida Straits

S0 = 1361.0
OMEGA = 7.2921e-5
FILE_RE = re.compile(r"gomb4_daily_(\d{4})_(\d{3})_(3z|2d)\.nc$")


@dataclass(frozen=True)
class Layout:
    """The feature tables of a pack: what each state and forcing channel is."""

    depths: tuple[float, ...]

    @property
    def nlev(self) -> int:
        return len(self.depths)

    @property
    def state_names(self) -> list[str]:
        return [f"{v}_{d:g}m" for v in LEVEL_VARS for d in self.depths] + list(SURFACE_VARS)

    @property
    def state_units(self) -> list[str]:
        return [UNITS[v] for v in LEVEL_VARS for _ in self.depths] + [UNITS[v] for v in SURFACE_VARS]

    @property
    def forcing_names(self) -> list[str]:
        return list(WIND_VARS) + list(CALENDAR)

    @property
    def forcing_units(self) -> list[str]:
        return [UNITS[v] for v in self.forcing_names]

    def level_channels(self, k: int) -> list[int]:
        return [i * self.nlev + k for i in range(len(LEVEL_VARS))]

    @property
    def surface_channels(self) -> list[int]:
        return list(range(len(LEVEL_VARS) * self.nlev, len(self.state_names)))


@dataclass(frozen=True)
class Plan:
    """Everything that decides a pack's content. One output folder holds exactly one plan."""

    root: str
    start: date
    end: date
    train_end: date
    depths: tuple[float, ...] = DEFAULT_DEPTHS
    stride: int = 1
    band: int = 20
    max_gap: int = 3
    topo: str = DEFAULT_TOPO

    @property
    def layout(self) -> Layout:
        return Layout(tuple(float(d) for d in self.depths))

    def record(self) -> dict:
        r = asdict(self)
        r.update(format=FORMAT, start=self.start.isoformat(), end=self.end.isoformat(),
                 train_end=self.train_end.isoformat(), depths=[float(d) for d in self.depths])
        return r

    def days(self) -> np.ndarray:
        return np.arange(np.datetime64(self.start), np.datetime64(self.end) + np.timedelta64(1, "D"), dtype="datetime64[D]")


@dataclass(frozen=True)
class Grid:
    """What the files say about space, read once from the first day and topography.
    Every field is already subsampled by the plan's stride; level_ocean is per selected level."""

    level_index: np.ndarray  # (nlev,) into the file's Depth axis
    lon: np.ndarray  # (nx,)
    lat: np.ndarray  # (ny,)
    ocean: np.ndarray  # (ny, nx) bool
    level_ocean: np.ndarray  # (nlev, ny, nx) bool
    static: np.ndarray  # (static_feature, ny, nx) float32, STATIC_UNITS order
    boundary: np.ndarray  # (ny, nx) bool
    fill: np.ndarray  # (state_feature,) float32, the value of every point that is not real

    def save(self, path: Path) -> None:
        tmp = path.with_name(path.name + ".partial")
        with open(tmp, "wb") as f:
            np.savez(f, **asdict(self))
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path) -> Grid:
        with np.load(path) as z:
            return cls(**{k: z[k] for k in z.files})


def day_paths(root: str | Path, day: np.datetime64) -> tuple[Path, Path]:
    d = day.astype(date)
    stem = Path(root) / f"{d.year:04d}" / f"gomb4_daily_{d.year:04d}_{d.timetuple().tm_yday:03d}"
    return Path(f"{stem}_3z.nc"), Path(f"{stem}_2d.nc")


def listing(root: str | Path, years) -> dict[str, set[np.datetime64]]:
    """{"3z": days, "2d": days} present on disk, from directory listings only."""
    found = {"3z": set(), "2d": set()}
    for y in years:
        ydir = Path(root) / f"{y:04d}"
        if not ydir.is_dir():
            continue
        for name in (p.name for p in ydir.iterdir()):
            m = FILE_RE.fullmatch(name)
            if m and int(m[1]) == y:
                found[m[3]].add(np.datetime64(f"{y:04d}-01-01") + np.timedelta64(int(m[2]) - 1, "D"))
    return found


def schedule(plan: Plan) -> tuple[np.ndarray, np.ndarray]:
    """(days, present). Raises when an end day is missing or a run of missing days exceeds max_gap."""
    days = plan.days()
    found = listing(plan.root, range(plan.start.year, plan.end.year + 1))
    present = np.array([d in found["3z"] and d in found["2d"] for d in days])
    if not (present[0] and present[-1]):
        raise ValueError(f"start and end must have both files: {days[0]} {present[0]}, {days[-1]} {present[-1]}")
    gaps = _runs(~present)
    long = [(days[a], days[b - 1]) for a, b in gaps if b - a > plan.max_gap]
    if long:
        raise ValueError(f"gaps longer than {plan.max_gap} days: " + ", ".join(f"{a}..{b}" for a, b in long))
    return days, present


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) of each run of True."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def _valid(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & (np.abs(a) < FILL)


def read_day(root: str | Path, day: np.datetime64, level_index, stride: int):
    """Raw (levels (var, level, y, x), surface (var, y, x), wind (var, y, x)) for one day, subsampled.
    Values are as stored: fill and NaN included."""
    import netCDF4  # only the archive readers need it (the skynet venv has none; refloor and zscan run there)

    p3, p2 = day_paths(root, day)
    s = slice(None, None, stride)
    top = int(max(level_index)) + 1
    with netCDF4.Dataset(p3) as nc:
        nc.set_auto_maskandscale(False)
        # One contiguous depth slice per variable: a list of depth indices, or a strided read, is 2-3x slower
        # on the product's [1, 14, 193, 263] chunks (measured on skynet).
        levels = np.stack([nc[v][0, :top, :, :][level_index][:, s, s] for v in LEVEL_VARS.values()])
    with netCDF4.Dataset(p2) as nc:
        nc.set_auto_maskandscale(False)
        surface = np.stack([nc[v][0, :, :][s, s] for v in SURFACE_VARS.values()])
        wind = np.stack([nc[v][0, :, :][s, s] for v in WIND_VARS.values()])
    return levels.astype(np.float32), surface.astype(np.float32), wind.astype(np.float32)


def level_validity(levels: np.ndarray) -> np.ndarray:
    """(level, y, x): every level variable is valid there."""
    return _valid(levels).all(axis=0)


def read_topo(path: str | Path, ny: int, nx: int) -> np.ndarray:
    """Bottom depth (m) from a HYCOM .a file, land 0."""
    d = np.fromfile(path, dtype=">f4", count=ny * nx)
    if d.size != ny * nx:
        raise ValueError(f"{path}: {d.size} values, need {ny}x{nx}")
    d = d.reshape(ny, nx).astype(np.float32)
    return np.where(np.isfinite(d) & (np.abs(d) < FILL), d, 0.0).astype(np.float32)


def gulf_mask(ocean: np.ndarray, lon2d: np.ndarray, lat2d: np.ndarray, sections, seed) -> np.ndarray:
    """The ocean cells 4-connected to seed once the section lines are cut out. Section cells are not Gulf."""
    lon, lat = lon2d[0, :], lat2d[:, 0]
    step = min(np.abs(np.diff(lon)).min(), np.abs(np.diff(lat)).min())
    cut = np.zeros(ocean.shape, bool)
    for (x0, y0), (x1, y1) in sections:
        n = max(2, int(np.ceil(4 * np.hypot(x1 - x0, y1 - y0) / step)) + 1)
        xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
        cut[_nearest(lat, ys), _nearest(lon, xs)] = True
    cut = ndimage.binary_dilation(cut)  # the cross structure: a staircase line cannot leak diagonally
    labels, _ = ndimage.label(ocean & ~cut)
    j, i = _nearest(lat, [seed[1]])[0], _nearest(lon, [seed[0]])[0]
    if labels[j, i] == 0:
        raise ValueError(f"gulf seed {seed} is not on an ocean cell")
    return labels == labels[j, i]


def _nearest(axis: np.ndarray, values) -> np.ndarray:
    return np.abs(np.asarray(values)[:, None] - axis[None, :]).argmin(axis=1)


def boundary_mask(ocean: np.ndarray, band: int) -> np.ndarray:
    """Land, and ocean within band cells of an ocean cell in the outer two rows or columns."""
    edge = np.ones(ocean.shape, bool)
    edge[2:-2, 2:-2] = False
    open_ = ocean & edge
    if not open_.any():
        return ~ocean
    return ~ocean | (ndimage.distance_transform_edt(~open_) <= band)


def make_grid(plan: Plan) -> Grid:
    """Static fields and masks from the start day and topography, computed on the native grid."""
    import netCDF4

    p3, _ = day_paths(plan.root, np.datetime64(plan.start))
    with netCDF4.Dataset(p3) as nc:
        nc.set_auto_maskandscale(False)
        axis = nc["Depth"][:].astype(float)
        lon, lat = nc["Longitude"][:].astype(float), nc["Latitude"][:].astype(float)
        ocean = _valid(nc["water_temp"][0, 0, :, :])
    bad = [d for d in plan.depths if not np.isclose(axis, d, atol=1e-3).any()]
    if bad:
        raise ValueError(f"depths not in the file's Depth axis: {bad}")
    level_index = np.array([int(np.argmin(np.abs(axis - d))) for d in plan.depths])
    levels, surface, _ = read_day(plan.root, np.datetime64(plan.start), level_index, plan.stride)
    level_ocean = level_validity(levels)
    fill = [levels[i, k][level_ocean[k]].mean() for i in range(len(LEVEL_VARS)) for k in range(len(plan.depths))]
    fill += [a[_valid(a)].mean() for a in surface]

    lon2d, lat2d = np.meshgrid(lon, lat)
    native = {
        "depth": read_topo(plan.topo, lat.size, lon.size),
        "lon": lon2d,
        "lat": lat2d,
        "coriolis": 2 * OMEGA * np.sin(np.deg2rad(lat2d)),
        "ocean": ocean,
        "gulf": gulf_mask(ocean, lon2d, lat2d, GULF_SECTIONS, GULF_SEED),
    }
    s = (slice(None, None, plan.stride),) * 2
    return Grid(
        level_index=level_index,
        lon=lon[s[1]],
        lat=lat[s[0]],
        ocean=ocean[s],
        level_ocean=level_ocean,
        static=np.stack([native[k][s] for k in STATIC_UNITS]).astype(np.float32),
        boundary=boundary_mask(ocean, plan.band)[s],
        fill=np.array(fill, np.float32),
    )


def calendar(day: np.datetime64, lat: np.ndarray, nx: int) -> np.ndarray:
    """(sin_doy, cos_doy, insolation) as (3, y, x): spatially uniform phase and daily-mean TOA insolation."""
    d = day.astype(date)
    doy = d.timetuple().tm_yday
    ndays = 366 if (d.year % 4 == 0 and d.year % 100 != 0) or d.year % 400 == 0 else 365
    phase = 2 * np.pi * (doy - 0.5) / ndays
    phi = np.deg2rad(lat)
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    h0 = np.arccos(np.clip(-np.tan(phi) * np.tan(decl), -1.0, 1.0))
    ecc = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    q = S0 / np.pi * ecc * (h0 * np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.sin(h0))
    shape = (lat.size, nx)
    return np.stack([np.full(shape, np.sin(phase)), np.full(shape, np.cos(phase)), np.broadcast_to(q[:, None], shape)])


def assemble(raw, day: np.datetime64, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """One row as (grid_index, state_feature) and (grid_index, forcing_feature), no fill values left."""
    levels, surface, wind = raw
    lost = grid.level_ocean & ~level_validity(levels)
    if lost[0].any() or lost.sum() > MAX_LOST_FRACTION * grid.level_ocean.sum():
        raise ValueError(f"{day}: {int(lost.sum())} points of the first day's ocean are missing ({int(lost[0].sum())} at the surface); a bad file?")
    levels = levels.copy()
    for k in np.flatnonzero(lost.any(axis=(1, 2))):
        levels[:, k][:, lost[k]] = levels[:, k - 1][:, lost[k]]
    state = np.concatenate([levels.reshape(-1, *grid.ocean.shape), surface])
    real = np.concatenate([np.tile(grid.level_ocean, (len(LEVEL_VARS), 1, 1)), _valid(surface)])
    state = np.where(real, state, grid.fill[:, None, None])
    forcing = np.concatenate([np.where(_valid(wind), wind, 0.0), calendar(day, grid.lat, grid.lon.size)])
    return _to_grid_index(state).astype(np.float32), _to_grid_index(forcing).astype(np.float32)


_WORKER: dict = {}


def _open_rows(out: Path, mode: str = "r+") -> dict[str, np.memmap]:
    return {k: np.lib.format.open_memmap(out / f"{k}.npy", mode=mode) for k in ("state", "forcing")}


def _init_worker(plan: Plan, grid: Grid, days: np.ndarray, out: Path) -> None:
    _WORKER.update(plan=plan, grid=grid, days=days, arrays=_open_rows(out))


def _write_row(n: int) -> tuple[int, float]:
    t0 = time.perf_counter()
    plan, grid, day, arrays = _WORKER["plan"], _WORKER["grid"], _WORKER["days"][n], _WORKER["arrays"]
    st, fo = assemble(read_day(plan.root, day, grid.level_index, plan.stride), day, grid)
    arrays["state"][n], arrays["forcing"][n] = st, fo
    for a in arrays.values():
        a.flush()
    return n, time.perf_counter() - t0


def build(plan: Plan, out: Path, workers: int = 32, dtype: str = "float32") -> None:
    if dtype not in ("float32", "int16"):
        raise ValueError(f"dtype {dtype!r}: use float32 or int16")
    out = Path(out)
    plan_file, meta = out / "plan.json", out / "meta.zarr"
    if plan_file.exists():
        old = json.loads(plan_file.read_text())
        if old != plan.record():
            changed = sorted(k for k in old.keys() | plan.record().keys() if old.get(k) != plan.record().get(k))
            raise ValueError(f"{out} holds another plan (differs in {changed}); use a new folder")
        if meta.is_dir() and np.load(out / "written.npy").all():
            stored = packed_dtype(out)
            if dtype == "float32" and stored == "int16":
                raise ValueError(f"{out} is an int16 pack; a float32 pack needs a new folder")
            if dtype == "int16" and stored != "int16":
                quantize(out)
            print(f"{out}: complete", flush=True)
            return
    days, present = schedule(plan)
    out.mkdir(parents=True, exist_ok=True)
    if not plan_file.exists():
        tmp = out / "plan.json.partial"
        tmp.write_text(json.dumps(plan.record(), indent=1))
        tmp.rename(plan_file)
    shutil.rmtree(meta, ignore_errors=True)

    grid_file = out / "grid.npz"
    if not grid_file.exists():
        make_grid(plan).save(grid_file)
    grid = Grid.load(grid_file)
    layout = plan.layout
    npoint = grid.ocean.size
    for name, nfeat in (("state", len(layout.state_names)), ("forcing", len(layout.forcing_names))):
        if not (out / f"{name}.npy").exists():
            np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=np.float32, shape=(days.size, npoint, nfeat)).flush()
    if not (out / "written.npy").exists():
        np.save(out / "written.npy", np.zeros(days.size, bool))
    written = np.lib.format.open_memmap(out / "written.npy", mode="r+")

    todo = [int(n) for n in np.flatnonzero(present & ~written)]
    t0 = time.perf_counter()
    secs = []
    if workers == 1:
        _init_worker(plan, grid, days, out)
        results = map(_write_row, todo)
        pool = None
    else:
        pool = multiprocessing.get_context("fork").Pool(workers, _init_worker, (plan, grid, days, out))
        results = pool.imap_unordered(_write_row, todo)
    try:
        for n, sec in results:
            written[n] = True
            written.flush()
            secs.append(sec)
            print(f"{days[n]} row {n} {sec:.1f}s", flush=True)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        _WORKER.clear()

    nfill = _fill_missing(out, days, present, written, grid)
    wall = time.perf_counter() - t0
    mean = f", {np.mean(secs):.1f}s per row" if secs else ""
    print(f"{len(secs)} rows read, {nfill} filled in {wall:.0f}s with {workers} workers{mean}", flush=True)
    _write_meta(plan, out, days, present, grid)
    if dtype == "int16":
        quantize(out)


def packed_dtype(out: Path) -> str:
    """int16 once quantize finished (meta.zarr attrs), else float32."""
    attrs = xr.open_zarr(Path(out) / "meta.zarr", consolidated=True).attrs
    return "int16" if "int16_scale" in attrs else "float32"


def quantize(out: Path, rows: int = 8) -> dict[str, int]:
    """Convert a complete float32 pack's state and forcing to int16 fixed point in place; resumes (an array already
    int16 is kept) and records scale, clip and clipped counts in meta.zarr's attrs last. Returns the clipped counts."""
    import zarr

    out = Path(out)
    meta = xr.open_zarr(out / "meta.zarr", consolidated=True).load()
    limit = INT16_SCALE * INT16_CLIP
    clipped = {}
    for name in PACKED:
        path = out / f"{name}.npy"
        src = np.load(path, mmap_mode="r")
        if src.dtype != np.int16:
            mean, std = (meta[f"{name}_{op}"].values.astype(np.float64) for op in ("mean", "std"))
            tmp = out / f"{name}.npy.int16.partial"
            dst = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.int16, shape=src.shape)
            for n in range(0, src.shape[0], rows):
                z = np.rint((np.asarray(src[n : n + rows], np.float64) - mean) / std * INT16_SCALE)
                dst[n : n + rows] = np.clip(z, -limit, limit).astype(np.int16)
            dst.flush()
            del dst, src
            tmp.replace(path)
        q = np.load(path, mmap_mode="r")
        clipped[name] = int(sum(np.count_nonzero(np.abs(q[n : n + rows].astype(np.int32)) >= limit) for n in range(0, q.shape[0], rows)))
        print(f"{name}: int16, {clipped[name]} values at the +-{INT16_CLIP} std clip", flush=True)
    group = zarr.open_group(out / "meta.zarr", mode="r+")
    group.attrs.update({"int16_scale": INT16_SCALE, "int16_clip": INT16_CLIP,
                        **{f"int16_clipped_{k}": v for k, v in clipped.items()}})
    zarr.consolidate_metadata(out / "meta.zarr")
    return clipped


PACKED = ("state", "forcing")


def _fill_missing(out: Path, days: np.ndarray, present: np.ndarray, written: np.memmap, grid: Grid) -> int:
    """Linear interpolation in time between the nearest real rows; calendar channels computed normally."""
    arrays = _open_rows(out)
    real = np.flatnonzero(present)
    nwind = len(WIND_VARS)
    todo = np.flatnonzero(~present & ~written)
    for n in todo:
        p, q = real[real < n][-1], real[real > n][0]
        w = (n - p) / (q - p)
        arrays["state"][n] = (1 - w) * arrays["state"][p] + w * arrays["state"][q]
        wind = (1 - w) * arrays["forcing"][p][:, :nwind] + w * arrays["forcing"][q][:, :nwind]
        cal = _to_grid_index(calendar(days[n], grid.lat, grid.lon.size))
        arrays["forcing"][n] = np.concatenate([wind, cal], axis=1)
        for a in arrays.values():
            a.flush()
        written[n] = True
        written.flush()
    return todo.size


def _statistics(layout: Layout, arrays, use: np.ndarray, ocean: np.ndarray, level_ocean: np.ndarray) -> dict:
    """Train statistics per mask group: each level's channels over that level's real points, surface state
    and forcing over ocean. Changes count only between consecutive rows that are both used."""
    groups = [(layout.level_channels(k), level_ocean[:, k]) for k in range(layout.nlev)]
    groups.append((layout.surface_channels, ocean))
    acc = {k: [_Moments() for _ in groups] for k in ("state", "diff")}
    forcing = _Moments()
    prev = None
    for n in range(use.size):
        if not use[n]:
            prev = None
            continue
        row = np.asarray(arrays["state"][n])
        for g, (ch, mask) in enumerate(groups):
            x = row[:, ch][mask]
            acc["state"][g].add(x)
            if prev is not None:
                acc["diff"][g].add(x - prev[:, ch][mask])
        forcing.add(np.asarray(arrays["forcing"][n])[ocean])
        prev = row

    nstate = len(layout.state_names)
    mean = {k: np.zeros(nstate, np.float32) for k in acc}
    raw = {k: np.zeros(nstate, np.float32) for k in acc}
    for k, moments in acc.items():
        for (ch, _), m in zip(groups, moments):
            mean[k][ch] = m.mean
            raw[k][ch] = np.sqrt(m.m2 / m.n)
    state_std, floor_attrs = floored_state_std(layout, raw["state"])
    return {
        "state_mean": mean["state"],
        "state_std": state_std,
        "state_std_raw": _safe(raw["state"]),
        "state_diff_mean": mean["diff"],
        "state_diff_std": _safe(np.maximum(raw["diff"], DIFF_STD_FLOOR * raw["state"])),
        "forcing_mean": forcing.mean,
        "forcing_std": forcing.std,
    }, floor_attrs


def state_std_floor(layout: Layout, raw: np.ndarray) -> dict[str, float]:
    """The floor of each level variable's state std, from the raw std of its column's channels."""
    n = layout.nlev
    return {v: float(max(STATE_STD_MIN[v], STATE_STD_MEDIAN_FRACTION * np.median(raw[i * n : (i + 1) * n])))
            for i, v in enumerate(LEVEL_VARS)}


def floored_state_std(layout: Layout, raw: np.ndarray) -> tuple[np.ndarray, dict]:
    """(state std with each level channel at least its variable's floor, the meta.zarr attrs that record it)."""
    floor = state_std_floor(layout, raw)
    per_channel = np.zeros(len(layout.state_names))
    for i, v in enumerate(LEVEL_VARS):
        per_channel[i * layout.nlev : (i + 1) * layout.nlev] = floor[v]
    std = _safe(np.maximum(raw, per_channel))
    floored = [name for name, r, f in zip(layout.state_names, raw, per_channel) if r < f]
    attrs = {"state_std_floor": floor, "state_std_floored": floored,
             "state_std_floor_rule": f"max({STATE_STD_MIN}, {STATE_STD_MEDIAN_FRACTION} x median column std)"}
    return std, attrs


def _write_meta(plan: Plan, out: Path, days: np.ndarray, present: np.ndarray, grid: Grid) -> None:
    layout = plan.layout
    ocean = grid.ocean.T.reshape(-1)
    level_ocean = _to_grid_index(grid.level_ocean)
    static = _to_grid_index(grid.static)
    use = present & (days <= np.datetime64(plan.train_end))
    stats, floor_attrs = _statistics(layout, _open_rows(out, "r"), use, ocean, level_ocean)
    g = "grid_index"
    gx, gy = np.meshgrid(grid.lon, grid.lat, indexing="ij")
    feature = {"state": "state_feature", "forcing": "forcing_feature"}
    ds = xr.Dataset(
        {
            **{k: ((feature[k.split("_")[0]],), v.astype(np.float32)) for k, v in stats.items()},
            "static_mean": (("static_feature",), static[ocean].mean(axis=0).astype(np.float32)),
            "static_std": (("static_feature",), _safe(static[ocean].std(axis=0))),
            "static": ((g, "static_feature"), static),
            "boundary_mask": ((g,), grid.boundary.T.reshape(-1).astype(np.int8)),
            "level_ocean": ((g, "level"), level_ocean.astype(np.int8)),
            "time_filled": (("time",), ~present),
        },
        coords={
            "time": days.astype("datetime64[ns]") + np.timedelta64(12, "h"),
            "x": ((g,), gx.reshape(-1)),
            "y": ((g,), gy.reshape(-1)),
            "level": np.array(layout.depths),
            "state_feature": layout.state_names,
            "forcing_feature": layout.forcing_names,
            "static_feature": list(STATIC_UNITS),
            "state_feature_units": (("state_feature",), layout.state_units),
            "forcing_feature_units": (("forcing_feature",), layout.forcing_units),
            "static_feature_units": (("static_feature",), list(STATIC_UNITS.values())),
        },
        attrs={"train_end": plan.train_end.isoformat(), "stride": plan.stride, "depths": list(layout.depths),
               "root": plan.root, "band": plan.band, "source": "GOMb0.04 reanalysis daily means", **floor_attrs},
    )
    for v in ds.variables.values():
        v.encoding.clear()
    tmp = out / "meta.zarr.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    ds.to_zarr(tmp, mode="w", consolidated=True)
    tmp.rename(out / "meta.zarr")


LINKED = ("state.npy", "forcing.npy", "plan.json", "written.npy", "grid.npz")


def refloor(src: Path, out: Path) -> list[str]:
    """A new pack folder `out` that links src's arrays and holds src's meta.zarr with the state std floored, for a pack
    built before the floor (src is not changed: other runs read its statistics). For an int16 pack the std its values
    were quantized with is kept as state_int16_std, which the datastore decodes with. Returns the floored channels."""
    src, out = Path(src).resolve(), Path(out)
    meta = xr.open_zarr(src / "meta.zarr", consolidated=True).load()
    if "state_std_floored" in meta.attrs:
        raise ValueError(f"{src} already has a floored state std")
    if out.exists():
        raise ValueError(f"{out} exists")
    layout = Layout(tuple(float(d) for d in meta.attrs["depths"]))
    raw = meta.state_std.values.astype(np.float64)
    std, attrs = floored_state_std(layout, raw)
    ds = meta.copy()
    if "int16_scale" in meta.attrs:
        ds["state_int16_std"] = meta.state_std
    ds["state_std_raw"] = meta.state_std
    ds["state_std"] = (("state_feature",), std.astype(np.float32))
    ds.attrs.update(attrs | {"refloored_from": str(src)})
    for v in ds.variables.values():
        v.encoding.clear()
    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name in LINKED:
        if (src / name).exists():
            (tmp / name).symlink_to(src / name)
    ds.to_zarr(tmp / "meta.zarr", mode="w", consolidated=True)
    tmp.rename(out)
    return attrs["state_std_floored"]


def zscan(pack: Path, every: int = 10) -> dict:
    """Largest standardized |state| per channel over every `every`-th row up to train_end, on the real points of each
    level and on the Gulf ones, with the raw std and with the floored one. {"rows", "channels": {name: {raw_std, std,
    max_raw_sigma, max_sigma, gulf_max_sigma, at: [lon, lat, day], gulf_at}}}."""
    from hycom_emulator.datastore import _open

    pack = Path(pack)
    meta = xr.open_zarr(pack / "meta.zarr", consolidated=True).load()
    layout = Layout(tuple(float(d) for d in meta.attrs["depths"]))
    raw = (meta["state_std_raw"] if "state_std_raw" in meta else meta.state_std).values.astype(np.float64)
    std, _ = floored_state_std(layout, raw)
    state = _open(pack).state
    t = meta.time.values.astype("datetime64[D]")
    rows = np.flatnonzero(t <= np.datetime64(meta.attrs["train_end"]))[::every]
    mean = meta.state_mean.values.astype(np.float64)
    level_ocean = meta.level_ocean.values.astype(bool)
    real = np.ones((level_ocean.shape[0], len(layout.state_names)), bool)
    for i in range(len(LEVEL_VARS)):
        real[:, i * layout.nlev : (i + 1) * layout.nlev] = level_ocean
    gulf = meta.static.sel(static_feature="gulf").values > 0.5
    masks = {"": real, "gulf_": real & gulf[:, None]}
    worst = {k: np.zeros(real.shape[1]) for k in masks}
    where = {k: np.zeros((real.shape[1], 2), int) for k in masks}
    for n in rows:
        anomaly = np.abs(np.asarray(state[n].values, np.float64) - mean)
        for k, mask in masks.items():
            a = np.where(mask, anomaly, 0.0)
            i = a.argmax(0)
            v = a[i, np.arange(a.shape[1])]
            better = v > worst[k]
            worst[k][better], where[k][better] = v[better], np.stack([i, np.full_like(i, n)], 1)[better]
    x, y = meta.x.values, meta.y.values
    at = lambda k, c: [float(x[where[k][c, 0]]), float(y[where[k][c, 0]]), str(t[where[k][c, 1]])]  # noqa: E731
    return {"rows": int(rows.size), "channels": {
        name: {"raw_std": float(raw[c]), "std": float(std[c]), "max_raw_sigma": float(worst[""][c] / raw[c]),
               "max_sigma": float(worst[""][c] / std[c]), "gulf_max_sigma": float(worst["gulf_"][c] / std[c]),
               "at": at("", c), "gulf_at": at("gulf_", c)}
        for c, name in enumerate(layout.state_names)}}


def inventory(root: str | Path, start: date | None = None, end: date | None = None) -> None:
    years = sorted(int(p.name) for p in Path(root).iterdir() if p.name.isdigit())
    years = [y for y in years if (start is None or y >= start.year) and (end is None or y <= end.year)]
    found = listing(root, years)
    lo = np.datetime64(start) if start else None
    hi = np.datetime64(end) if end else None
    for y in years:
        days = np.arange(np.datetime64(f"{y:04d}-01-01"), np.datetime64(f"{y + 1:04d}-01-01"), dtype="datetime64[D]")
        days = days[(days >= lo if lo is not None else True) & (days <= hi if hi is not None else True)]
        n3, n2 = (sum(d in found[k] for d in days) for k in ("3z", "2d"))
        missing = [str(d) for d in days if d not in found["3z"] or d not in found["2d"]]
        print(f"{y}  3z {n3:3d}  2d {n2:3d}  days {days.size:3d}  missing {len(missing)}: {' '.join(missing)}")


def parse_levels(text: str) -> tuple[float, ...]:
    if text == "all":
        return tuple(float(d) for d in DEPTH_AXIS)
    return tuple(sorted({float(d) for d in text.split(",")}))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--root", default=DEFAULT_ROOT)
    inv.add_argument("--start", type=date.fromisoformat)
    inv.add_argument("--end", type=date.fromisoformat)
    b = sub.add_parser("build")
    b.add_argument("out", type=Path)
    b.add_argument("--start", required=True, type=date.fromisoformat)
    b.add_argument("--end", required=True, type=date.fromisoformat)
    b.add_argument("--train-end", required=True, type=date.fromisoformat)
    b.add_argument("--root", default=DEFAULT_ROOT)
    b.add_argument("--topo", default=DEFAULT_TOPO)
    b.add_argument("--levels", type=parse_levels, default=DEFAULT_DEPTHS, help="comma-separated metres, or all")
    b.add_argument("--stride", type=int, default=1)
    b.add_argument("--band", type=int, default=20)
    b.add_argument("--max-gap", type=int, default=3)
    b.add_argument("--workers", type=int, default=32, help="parallel readers; keep under 64 on the shared server")
    b.add_argument("--dtype", choices=("float32", "int16"), default="float32",
                   help="int16: standardised fixed point (half the size), converted after the float32 build")
    r = sub.add_parser("refloor", help="a new folder linking a pre-floor pack's arrays, with the floored state std")
    r.add_argument("src", type=Path)
    r.add_argument("out", type=Path)
    z = sub.add_parser("zscan", help="largest standardized |state| per channel over train rows, raw and floored std")
    z.add_argument("pack", type=Path)
    z.add_argument("--every", type=int, default=10)
    z.add_argument("--json", type=Path)
    a = p.parse_args()
    if a.cmd == "refloor":
        print("floored:", " ".join(refloor(a.src, a.out)))
        return
    if a.cmd == "zscan":
        res = zscan(a.pack, a.every)
        if a.json:
            a.json.write_text(json.dumps(res, indent=1))
        print(f"{res['rows']} train rows")
        for name, c in res["channels"].items():
            print(f"{name:12s} std {c['raw_std']:.4g} -> {c['std']:.4g}  max {c['max_raw_sigma']:6.1f} -> {c['max_sigma']:6.1f} "
                  f"sigma at {c['at']}  Gulf {c['gulf_max_sigma']:6.1f} at {c['gulf_at']}")
        return
    if a.cmd == "inventory":
        inventory(a.root, a.start, a.end)
        return
    plan = Plan(root=a.root, start=a.start, end=a.end, train_end=a.train_end, depths=tuple(float(d) for d in a.levels),
                stride=a.stride, band=a.band, max_gap=a.max_gap, topo=a.topo)
    build(plan, a.out, a.workers, a.dtype)


if __name__ == "__main__":
    main()
