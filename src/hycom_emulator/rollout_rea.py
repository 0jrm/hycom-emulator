"""Long free runs of the reanalysis emulator from many truth starts, kept as per-(start, lead) statistics.

    python -m hycom_emulator.rollout_rea run <nlam.yaml> <ckpt> <out_dir> [--horizon 15] [--chunk 15] [--batch 8]
        [--starts 2022-01-01,...] [--stride-days N] [--limit N]
    python -m hycom_emulator.rollout_rea check <nlam.yaml> <ckpt> <out.json> --starts ... --horizon H --chunk C
    python -m hycom_emulator.rollout_rea figures <stats.nc> <out_dir>
    python -m hycom_emulator.rollout_rea movie <nlam.yaml> <ckpt> <out_dir> --starts 2022-01-01,2022-07-01,2023-01-01

The model unrolls in chunks of --chunk days. Each chunk is one neural-lam window (WeatherDataset over the whole
record: the config's train split is widened in memory) whose two initial states are replaced by the model's last
two predictions, so no truth enters the interior after t0. ARForecaster still overwrites the boundary band with
truth every step, and the forcing is the true daily wind.

Default starts: every day 2022-01-01..2024-08-30 (out of sample) and every 30 days 2001-02-01..2021-12-01 (in
sample), each run for --horizon days or up to the end of the record. `run` writes out/parts/<first-start>_<n>.nc per
batch of starts, skips starts already in parts on a rerun, and concatenates the parts into out/stats.nc with dims
(start, lead). STATS below is the single list of its variables. `check` measures how much chunking changes the raw
predictions against the run-to-run noise of a repeat. `figures` plots stats.nc; `movie` reruns a few starts and
animates SSH and T at 100 m.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import mmap
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import torch
import xarray as xr
from scipy import ndimage

import hycom_emulator.ensemble  # noqa: F401  registers crps_graph_lam, so a free run can load an ensemble checkpoint
from hycom_emulator.evaluate_rea import LEVEL_VARS, level_weights, point_weights

LABEL = ("regional free run: true boundary band (neural-lam overwrites it with truth every step) and true daily wind; "
         "no truth enters the interior after t0")
REGION = ("Gulf of Mexico (static gulf) ocean points outside the boundary band; area weight cos^2(lat), each channel "
          "on its real points (level_ocean); T, S, u, v as columns to 2000 m weighted by depth interval")
FIELDS = (*LEVEL_VARS, "ssh")
UNITS = {"temp": "degC", "salin": "psu", "u": "m/s", "v": "m/s", "ssh": "m"}
MEANS = ("temp_0m", "temp_100m", "salin_0m", "ssh")
BANDS = {"lt50km": (0, 50), "50to100km": (50, 100), "100to200km": (100, 200), "gt200km": (200, np.inf)}  # wavelength
SPECTRA = ("ssh", "temp_100m")
LC_LEVEL, LC_MIN_DEPTH, LCE_MIN_KM2 = 0.17, 500.0, 1000.0  # gom-da runs/emu-align-probe7/probe.py
YUCATAN_STRIP = {"lat": (21.9, 22.4), "lon": (-87.0, -84.8)}
IN_SAMPLE_END = np.datetime64("2022-01-01")
DAY = np.timedelta64(1, "D")


@dataclass(frozen=True)
class Start:
    date: np.datetime64  # t0, datetime64[D]
    index: int  # row of t0 in the record
    horizon: int  # days to run


def max_horizon(s: int, T: int) -> int:
    """Longest run from row s of T: the last target needs one more forcing row, and row s-1 is the second init."""
    return T - 2 - s if s >= 1 else 0


def window_index(s: int, c: int, chunk: int) -> int:
    """WeatherDataset index of chunk c of a run from row s: init rows idx, idx+1 = s-1+c*chunk, s+c*chunk."""
    return s - 1 + c * chunk


def start_set(dates: np.ndarray, horizon: int, starts: list[str] | None = None, stride: int = 1,
              limit: int | None = None) -> list[Start]:
    """Starts ordered by horizon (longest first), then date. dates: the record's row dates, datetime64[D]."""
    index = {d: i for i, d in enumerate(dates)}
    if starts is None:
        ins = np.arange(np.datetime64("2001-02-01"), np.datetime64("2021-12-02"), 30 * DAY)
        oos = np.arange(IN_SAMPLE_END, np.datetime64("2024-08-31"), stride * DAY)
        wanted = [d for d in (*ins, *oos) if d in index]
    else:
        wanted = [np.datetime64(s, "D") for s in starts]
        if missing := [str(d) for d in wanted if d not in index]:
            raise ValueError(f"starts outside the record: {', '.join(missing)}")
    out = [Start(d, index[d], min(horizon, max_horizon(index[d], len(dates)))) for d in wanted]
    return sorted((s for s in out if s.horizon >= 1), key=lambda s: (-s.horizon, s.date))[:limit]


@dataclass(frozen=True)
class Geometry:
    names: list[str]
    groups: dict  # field -> (column indices, (grid, k) float64 weights: area x region x level_ocean x dz)
    point: torch.Tensor  # (grid, feature) float64, evaluate_rea.point_weights
    region: torch.Tensor  # (grid,) bool
    area_sum: float  # sum of area over region
    plane: Plane | None = None  # 2D diagnostics (spectra, Loop Current); load_model sets it


@dataclass(frozen=True)
class Plane:
    """The grid as (y, x) images, field.reshape(nx, ny).T, for the spectra and the Loop Current."""
    shape: tuple[int, int]  # (nx, ny)
    box: tuple[slice, slice]  # (y, x) bounding box of the spectral masks
    taper: dict  # channel -> (by, bx) float64 tensor: its region mask x 2D Hann window, on the box
    band: torch.Tensor  # (by, bx) long: BANDS index of each FFT bin by wavelength, -1 for the mean
    lat: np.ndarray  # (ny, nx)
    cell_km2: np.ndarray  # (ny, nx) Mercator cell area
    gulf500: np.ndarray  # (ny, nx) bool: Gulf deeper than LC_MIN_DEPTH
    strip: np.ndarray  # (ny, nx) bool: the Yucatan strip seed inside gulf500


def plane(shape: tuple[int, int], lon: np.ndarray, lat: np.ndarray, depth: np.ndarray, gulf: np.ndarray,
          masks: dict[str, np.ndarray], stride: int, device="cpu") -> Plane:
    """lon, lat, depth, gulf and masks (channel -> region where it is scored) are (grid,) arrays."""
    img = lambda v: v.reshape(shape).T  # noqa: E731
    lon2, lat2 = img(lon), img(lat)
    side = 0.04 * stride * 111.32 * np.cos(np.deg2rad(lat2))  # km
    union = np.any([img(m) for m in masks.values()], 0)
    ys, xs = np.nonzero(union)
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    window = np.outer(np.hanning(box[0].stop - box[0].start), np.hanning(box[1].stop - box[1].start))
    dx = float(side[union].mean())
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(window.shape[0], dx), np.fft.fftfreq(window.shape[1], dx), indexing="ij"))
    wavelength = np.divide(1.0, k, out=np.full_like(k, np.inf), where=k > 0)
    band = np.full(k.shape, -1)
    for i, (lo, hi) in enumerate(BANDS.values()):
        band[(k > 0) & (wavelength >= lo) & (wavelength < hi)] = i
    gulf500 = img(gulf) & (img(depth) > LC_MIN_DEPTH)
    strip = (gulf500 & (lat2 >= YUCATAN_STRIP["lat"][0]) & (lat2 <= YUCATAN_STRIP["lat"][1])
             & (lon2 >= YUCATAN_STRIP["lon"][0]) & (lon2 <= YUCATAN_STRIP["lon"][1]))
    t = functools.partial(torch.as_tensor, device=device)
    return Plane(shape, box, {c: t(img(m)[box] * window, dtype=torch.float64) for c, m in masks.items()}, t(band),
                 lat2, side**2, gulf500, strip)


def band_variance(x: torch.Tensor, pl: Plane, ch: str) -> torch.Tensor:
    """(B, band) variance of x (B, grid) per wavelength band over the channel's region, after removing its mean
    and tapering with a Hann window; the bands sum to the tapered variance (Parseval)."""
    w = pl.taper[ch]
    f = x.double().reshape(x.shape[0], *pl.shape).transpose(1, 2)[:, pl.box[0], pl.box[1]]
    a = (f - (f * w).sum((1, 2), keepdim=True) / w.sum()) * w
    p = torch.fft.fft2(a).abs() ** 2 / (w.numel() * (w * w).sum())
    return torch.stack([p[:, pl.band == i].sum(1) for i in range(len(BANDS))], 1)


def loop_current(ssh: np.ndarray, pl: Plane) -> tuple[float, int, float]:
    """(northern extent in degrees N, LCE count, LC area km2) of one (ny, nx) SSH image, as probe7 defines them:
    SSH minus its mean over Gulf water deeper than 500 m, the 0.17 m region of that water holding the Yucatan
    strip seed, and eddies = the other regions of at least 1000 km2."""
    m, w = pl.gulf500, pl.cell_km2
    hi = m & (ssh - (ssh[m] * w[m]).sum() / w[m].sum() >= LC_LEVEL)
    lab, n = ndimage.label(hi)
    seed = lab[pl.strip & hi]
    k = int(np.bincount(seed[seed > 0]).argmax()) if (seed > 0).any() else 0
    areas = ndimage.sum(w, lab, index=np.arange(1, n + 1))
    n_lce = sum(1 for i in range(n) if i + 1 != k and areas[i] >= LCE_MIN_KM2)
    return (float(pl.lat[lab == k].max()), n_lce, float(areas[k - 1])) if k else (np.nan, n_lce, 0.0)


def geometry(names: list[str], area: np.ndarray, region: np.ndarray, level_ocean: np.ndarray, levels: np.ndarray,
             device="cpu") -> Geometry:
    pw = point_weights(names, area, region, level_ocean, levels)
    dz = level_weights(levels)
    groups = {}
    for f in LEVEL_VARS:
        cols = [j for j, n in enumerate(names) if n.rpartition("_")[0] == f]
        assert [float(names[j].rpartition("_")[2][:-1]) for j in cols] == levels.tolist(), f"{f} channels out of level order"
        groups[f] = (cols, pw[:, cols] * dz)
    groups["ssh"] = ([names.index("ssh")], pw[:, [names.index("ssh")]])
    t = functools.partial(torch.as_tensor, device=device)
    return Geometry(names, {f: (t(c), t(w, dtype=torch.float64)) for f, (c, w) in groups.items()},
                    t(pw, dtype=torch.float64), t(region), float((area * region).sum()))


@dataclass
class Step:
    """One lead of a batch of runs, raw units, (B, grid, feature). x0 is the truth at t0, and the previous model
    and truth states at lead 1 are x0."""
    pred: torch.Tensor
    truth: torch.Tensor
    x0: torch.Tensor
    pred_prev: torch.Tensor
    truth_prev: torch.Tensor
    cache: dict = field(default_factory=dict)  # per-step results that several STATS share


def _err(s: Step, g: Geometry, f: str, ref: str):
    cols, w = g.groups[f]
    return (getattr(s, ref)[..., cols] - s.truth[..., cols]).double(), w


def rmse(s: Step, g: Geometry, f: str, ref: str):
    e, w = _err(s, g, f, ref)
    return ((w * e * e).sum((1, 2)) / w.sum()).sqrt()


def bias(s: Step, g: Geometry, f: str, ref: str):
    e, w = _err(s, g, f, ref)
    return (w * e).sum((1, 2)) / w.sum()


def corr_change(s: Step, g: Geometry, f: str):
    cols, w = g.groups[f]
    a = (s.pred[..., cols] - s.pred_prev[..., cols]).double()
    b = (s.truth[..., cols] - s.truth_prev[..., cols]).double()
    a = a - (w * a).sum((1, 2), keepdim=True) / w.sum()
    b = b - (w * b).sum((1, 2), keepdim=True) / w.sum()
    den = ((w * a * a).sum((1, 2)) * (w * b * b).sum((1, 2))).sqrt()
    return torch.where(den > 0, (w * a * b).sum((1, 2)) / den, torch.nan)


def gulf_mean(s: Step, g: Geometry, ch: str, which: str):
    w = g.point[:, g.names.index(ch)]
    return (w * getattr(s, which)[..., g.names.index(ch)].double()).sum(1) / w.sum()


def ke(s: Step, g: Geometry, which: str):
    x = getattr(s, which)
    (cu, w), (cv, _) = g.groups["u"], g.groups["v"]
    return (w * 0.5 * (x[..., cu].double() ** 2 + x[..., cv].double() ** 2)).sum((1, 2)) / g.area_sum


def maxspeed(s: Step, g: Geometry, which: str):
    x = getattr(s, which)
    speed = torch.hypot(x[..., g.names.index("u_0m")], x[..., g.names.index("v_0m")])
    return speed[:, g.region].amax(1).double()


def _cached(s: Step, key, compute):
    if key not in s.cache:
        s.cache[key] = compute()
    return s.cache[key]


def spectrum(s: Step, g: Geometry, ch: str, which: str, band: int):
    def var(w):
        return _cached(s, ("spec", ch, w), lambda: band_variance(getattr(s, w)[..., g.names.index(ch)], g.plane, ch))
    v = var("pred") / var("truth") if which == "ratio" else var(which)
    return v[:, band]


def lc(s: Step, g: Geometry, which: str, item: int):
    def rows(w):
        def compute():
            x = getattr(s, w)[..., g.names.index("ssh")].cpu().numpy()
            return torch.tensor([loop_current(r.reshape(g.plane.shape).T, g.plane) for r in x], dtype=torch.float64, device=s.pred.device)
        return _cached(s, ("lc", w), compute)
    v = rows("pred") - rows("truth") if which == "error" else rows(which)
    return v[:, item]


class Stat(NamedTuple):
    fn: Callable  # (Step, Geometry) -> (B,) tensor
    units: str
    long_name: str


def _registry() -> dict[str, Stat]:
    p = functools.partial
    st = {}
    for f in FIELDS:
        what = f"{f} column to 2000 m" if f in LEVEL_VARS else f
        for ref, tag, who in (("pred", "", "model"), ("x0", "pers_", "persistence of the start state")):
            st[f"rmse_{tag}{f}"] = Stat(p(rmse, f=f, ref=ref), UNITS[f], f"RMSE of {what}, {who}")
            st[f"bias_{tag}{f}"] = Stat(p(bias, f=f, ref=ref), UNITS[f], f"bias of {what}, {who} minus truth")
        st[f"corr_change_{f}"] = Stat(p(corr_change, f=f), "1", f"correlation of model and true 1-day change of {what}")
    for ch in MEANS:
        for which, who in (("pred", "model"), ("truth", "truth")):
            st[f"mean_{ch}_{who}"] = Stat(p(gulf_mean, ch=ch, which=which), UNITS[ch.split("_")[0]], f"Gulf mean {ch}, {who}")
    for which, who in (("pred", "model"), ("truth", "truth")):
        st[f"ke_{who}"] = Stat(p(ke, which=which), "m3 s-2", f"Gulf mean column kinetic energy to 2000 m per unit density, {who}")
        st[f"maxspeed_{who}"] = Stat(p(maxspeed, which=which), "m/s", f"Gulf maximum surface speed, {who}")
    for ch in SPECTRA:
        units = UNITS[ch.split("_")[0]]
        for b, (name, (lo, hi)) in enumerate(BANDS.items()):
            for which, u, who in (("pred", f"{units}2", "model"), ("truth", f"{units}2", "truth"), ("ratio", "1", "ratio model/truth")):
                st[f"spec_{ch}_{name}_{who.split()[0]}"] = Stat(p(spectrum, ch=ch, which=which, band=b), u,
                                                               f"variance of {ch} at wavelengths {lo}-{hi} km over the Gulf (Hann taper), {who}")
    for item, (key, units, what) in enumerate((("north", "degrees_north", "northern extent"), ("lce", "1", "eddy (LCE) count"),
                                               ("area", "km2", "area"))):
        for which, who in (("pred", "model"), ("truth", "truth"), ("error", "error")):
            if key == "north" or which != "error":
                st[f"lc_{key}_{who}"] = Stat(p(lc, which=which, item=item), units,
                                             f"Loop Current {what} from the 17 cm contour (probe7), {who if who != 'error' else 'model minus truth'}")
    return st


STATS = _registry()


def _pad(t, n: int):
    """Repeat the last step up to n steps; the unroll is causal, so the padded steps never change the valid ones."""
    return torch.cat([t, t[-1:].repeat(n - len(t), *[1] * (t.dim() - 1))])


def unroll(module, window: Callable, times: np.ndarray, starts: list[Start], chunk: int):
    """Yield (lead, positions in starts, Step) for every lead of every run, lead by lead. window(n) is a
    WeatherDataset with ar_steps=n over the whole record; times its row times (datetime64[ns])."""
    alive = list(range(len(starts)))
    carry = None
    for c in range(-(-max(s.horizon for s in starts) // chunk)):
        keep = [i for i, b in enumerate(alive) if starts[b].horizon > c * chunk]
        alive = [alive[i] for i in keep]
        steps = [min(chunk, starts[b].horizon - c * chunk) for b in alive]
        n = max(steps)
        samples = []
        for b, k in zip(alive, steps):
            s = starts[b]
            sample = window(k)[window_index(s.index, c, chunk)]
            r = s.index + c * chunk + 1
            assert (sample[3].numpy() == times[r:r + k].astype("datetime64[ns]").astype(np.int64)).all(), f"{s.date} chunk {c}"
            samples.append(sample)
        target, forcing, ttimes = (torch.stack([_pad(x[i], n) for x in samples]).to(module.device) for i in (1, 2, 3))
        if carry is None:
            carry = torch.stack([x[0] for x in samples]).to(module.device)
            x0 = truth_last = carry[:, 1]
        else:
            carry, x0, truth_last = carry[keep], x0[keep], truth_last[keep]
        with torch.no_grad():
            batch = module.on_after_batch_transfer((carry, target, forcing, ttimes), 0)
            pred = module.common_step(batch)[0] * module.state_std + module.state_mean
        for l in range(n):
            rows = torch.tensor([i for i, k in enumerate(steps) if l < k], device=module.device)
            pp, tp = (carry[:, 1], truth_last) if l == 0 else (pred[:, l - 1], target[:, l - 1])
            yield c * chunk + l + 1, [alive[i] for i in rows.tolist()], Step(pred[rows, l], target[rows, l], x0[rows], pp[rows], tp[rows])
        carry, truth_last = torch.cat([carry, pred[:, -2:]], 1)[:, -2:], target[:, -1]


def rollout_stats(module, window, times, starts: list[Start], chunk: int, geo: Geometry) -> np.ndarray:
    """(start, lead, stat) values in STATS order; NaN beyond each start's horizon."""
    out = np.full((len(starts), max(s.horizon for s in starts), len(STATS)), np.nan)
    for lead, pos, step in unroll(module, window, times, starts, chunk):
        out[pos, lead - 1] = torch.stack([st.fn(step, geo) for st in STATS.values()], 1).cpu().numpy()
    return out


def to_dataset(starts: list[Start], values: np.ndarray, attrs: dict) -> xr.Dataset:
    start = np.array([s.date for s in starts], "datetime64[ns]")
    lead = np.arange(1, values.shape[1] + 1)
    data = {k: (("start", "lead"), values[..., i], {"units": st.units, "long_name": st.long_name}) for i, (k, st) in enumerate(STATS.items())}
    coords = {"start": start, "lead": ("lead", lead, {"long_name": "lead (days)"}), "in_sample": ("start", start < IN_SAMPLE_END),
              "horizon": ("start", np.array([s.horizon for s in starts]))}
    return _with_valid(xr.Dataset(data, coords, attrs))


def _with_valid(ds: xr.Dataset) -> xr.Dataset:
    return ds.assign_coords(valid=(("start", "lead"), ds.start.values[:, None] + ds.lead.values * DAY))


def _atomic_netcdf(ds: xr.Dataset, path: Path) -> None:
    tmp = path.with_name(path.name + ".partial")
    ds.to_netcdf(tmp)
    os.replace(tmp, path)


def write_part(parts: Path, ds: xr.Dataset) -> Path:
    path = parts / f"{str(ds.start.values[0])[:10]}_{ds.sizes['start']}.nc"
    _atomic_netcdf(ds, path)
    return path


def done(parts: Path) -> dict[np.datetime64, tuple[int, str]]:
    """Start date -> (horizon, checkpoint md5) of every start already in parts."""
    out = {}
    for p in sorted(parts.glob("*.nc")):
        with xr.open_dataset(p) as d:
            for t, h in zip(d.start.values.astype("datetime64[D]"), d.horizon.values):
                out[t] = (int(h), d.attrs["checkpoint_md5"])
    return out


def finalize(parts: Path, out: Path) -> xr.Dataset:
    ds = xr.concat([xr.load_dataset(p).drop_vars("valid") for p in sorted(parts.glob("*.nc"))], "start", join="outer",
                   combine_attrs="override")
    ds = _with_valid(ds.sortby("start"))
    _atomic_netcdf(ds, out)
    return ds


@dataclass
class Model:
    module: object
    window: Callable
    times: np.ndarray  # row times, datetime64[ns]
    geo: Geometry
    meta: xr.Dataset
    shape: tuple[int, int]  # (nx, ny); grid_index = ix * ny + iy
    datastore: object


def load_model(config: Path, ckpt: Path) -> Model:
    from neural_lam.weather_dataset import WeatherDataset

    from hycom_emulator.evaluate_b00 import load

    ds, _, module = load(config, ckpt, "train", 1)
    ds.config["splits"]["train"] = ["1900-01-01", "2200-12-31"]
    meta = xr.open_zarr(Path(ds.config["zarr"]) / "meta.zarr", consolidated=True)
    names = [str(n) for n in meta.state_feature.values]
    assert names == ds.get_vars_names("state"), "datastore and meta.zarr disagree on state features"
    static = meta.static.load()
    region = (static.sel(static_feature="gulf").values.astype(bool) & static.sel(static_feature="ocean").values.astype(bool)
              & ~meta.boundary_mask.values.astype(bool))
    area = np.cos(np.deg2rad(static.sel(static_feature="lat").values)) ** 2
    levels = meta.level.values.astype(float)
    level_ocean = meta.level_ocean.values.astype(bool)
    shape = (ds.grid_shape_state.x, ds.grid_shape_state.y)
    st = lambda f: static.sel(static_feature=f).values  # noqa: E731
    masks = {"ssh": region, "temp_100m": region & level_ocean[:, int(np.argmin(np.abs(levels - 100)))]}
    pl = plane(shape, st("lon"), st("lat"), st("depth"), st("gulf").astype(bool), masks, int(meta.attrs["stride"]), module.device)
    geo = replace(geometry(names, area, region, level_ocean, levels, module.device), plane=pl)
    window = functools.cache(lambda n: WeatherDataset(ds, split="train", ar_steps=n, num_past_forcing_steps=1, num_future_forcing_steps=1))
    return Model(module, window, ds.get_dataarray("state", "train").time.values, geo, meta, shape, ds)


def release_pack_pages(datastore) -> None:
    """Unmap the pack rows read so far; they stay in the page cache. Without this RssFile grows with every row
    touched, up to the whole pack (157 GB at stride 2)."""
    for name in ("state", "forcing"):
        if isinstance(data := datastore._ds[name].data, np.memmap):
            data._mmap.madvise(mmap.MADV_DONTNEED)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _code_sha() -> str:
    if sha := os.environ.get("HYCOM_EMULATOR_SHA"):
        return sha
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def _rss_gib() -> dict[str, float]:
    f = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    return {k: int(f[k].split()[0]) / 2**20 for k in ("VmRSS", "RssAnon", "RssFile")}


def _gpu_gib() -> float:
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def run(config: Path, ckpt: Path, out: Path, horizon: int, chunk: int, batch: int, starts: list[str] | None,
        stride: int, limit: int | None) -> None:
    m = load_model(config, ckpt)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    md5 = _md5(ckpt)
    todo = start_set(m.times.astype("datetime64[D]"), horizon, starts, stride, limit)
    have = done(parts)
    if clash := [str(s.date) for s in todo if s.date in have and have[s.date] != (s.horizon, md5)]:
        raise ValueError(f"{parts} holds other runs (horizon or checkpoint differ) for {', '.join(clash[:5])}; use a new out_dir")
    todo = [s for s in todo if s.date not in have]
    attrs = {"label": LABEL, "checkpoint": str(ckpt), "checkpoint_md5": md5, "code_sha": _code_sha(), "region": REGION,
             "chunk": chunk, "max_horizon": horizon, "batch": batch,
             "interpolated_truth_days": " ".join(str(t)[:10] for t in m.times[m.meta.time_filled.values])}
    timing = out / "timing.json"
    total = json.loads(timing.read_text()) if timing.exists() else {"sample_steps": 0, "seconds": 0.0, "peak_gpu_gib": 0.0, "peak_rss_gib": 0.0}
    print(f"{len(todo)} starts to run, {len(have)} already in {parts}", flush=True)
    for i in range(0, len(todo), batch):
        group = todo[i:i + batch]
        t0 = time.perf_counter()
        values = rollout_stats(m.module, m.window, m.times, group, chunk, m.geo)
        attrs["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_part(parts, to_dataset(group, values, attrs))
        release_pack_pages(m.datastore)
        sec, n = time.perf_counter() - t0, sum(s.horizon for s in group)
        rss, gpu = _rss_gib(), _gpu_gib()
        total = {"sample_steps": total["sample_steps"] + n, "seconds": total["seconds"] + sec,
                 "peak_gpu_gib": max(total["peak_gpu_gib"], gpu), "peak_rss_gib": max(total["peak_rss_gib"], rss["VmRSS"])}
        timing.write_text(json.dumps(total, indent=1))
        print(f"starts {group[0].date}..{group[-1].date} ({len(group)}) steps {max(s.horizon for s in group)} {sec:.0f}s "
              f"{sec / n:.3f}s/sample-step gpu {gpu:.1f}GiB " + " ".join(f"{k} {v:.1f}GiB" for k, v in rss.items()), flush=True)
    ds = finalize(parts, out / "stats.nc")
    print(f"{out / 'stats.nc'}: {ds.sizes['start']} starts, leads 1..{ds.sizes['lead']}", flush=True)


def _channel_groups(names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for j, n in enumerate(names):
        groups.setdefault(n.rpartition("_")[0] if n.rpartition("_")[0] in LEVEL_VARS else n, []).append(j)
    return groups


def check(config: Path, ckpt: Path, out: Path, starts: list[str], horizon: int, chunk: int) -> dict:
    """Max abs and relative (to the single run's max |value|) differences of raw predictions per field group:
    chunked vs single window, and a repeat of the single window vs itself, the GPU noise floor."""
    m = load_model(config, ckpt)
    todo = start_set(m.times.astype("datetime64[D]"), horizon, starts)
    groups = _channel_groups(m.geo.names)
    acc = {g: np.zeros(3) for g in groups}
    runs = (unroll(m.module, m.window, m.times, todo, c) for c in (horizon, horizon, chunk))
    for (l1, p1, single), (l2, p2, repeat), (l3, p3, chunked) in zip(*runs):
        assert l1 == l2 == l3 and p1 == p2 == p3
        for g, cols in groups.items():
            ref = single.pred[..., cols]
            new = [(chunked.pred[..., cols] - ref).abs().max().item(), (repeat.pred[..., cols] - ref).abs().max().item(), ref.abs().max().item()]
            acc[g] = np.maximum(acc[g], new)
    res = {"label": LABEL, "checkpoint": str(ckpt), "starts": [str(s.date) for s in todo], "horizon": horizon, "chunk": chunk,
           "groups": {g: {"chunked_vs_single": {"max_abs": a[0], "max_rel": a[0] / a[2]},
                          "repeat_vs_single": {"max_abs": a[1], "max_rel": a[1] / a[2]}} for g, a in acc.items()}}
    out.write_text(json.dumps(res, indent=1))
    return res


def _band(ax, da: xr.DataArray, label: str, color: str) -> None:
    q = da.quantile([0.1, 0.5, 0.9], "start", skipna=True)
    ax.fill_between(da.lead, q.sel(quantile=0.1), q.sel(quantile=0.9), color=color, alpha=0.25, lw=0)
    ax.plot(da.lead, q.sel(quantile=0.5), color=color, label=label)


def figures(stats: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    ds = xr.load_dataset(stats)
    for tag, sel in (("out_of_sample", ~ds.in_sample), ("in_sample", ds.in_sample)):
        d = ds.sel(start=sel)
        if d.sizes["start"] == 0:
            continue
        span = f"{tag.replace('_', ' ')} starts {str(d.start.values[0])[:10]}..{str(d.start.values[-1])[:10]} (n={d.sizes['start']})"
        panels = {
            "rmse": [(f, [(d[f"rmse_{f}"], "model"), (d[f"rmse_pers_{f}"], "persistence")]) for f in FIELDS],
            "bias": [(f, [(d[f"bias_{f}"], "model"), (d[f"bias_pers_{f}"], "persistence")]) for f in FIELDS],
            "drift": [(ch, [(d[f"mean_{ch}_model"] - d[f"mean_{ch}_truth"], "model - truth")]) for ch in MEANS],
            "spectra": [(f"{ch} variance ratio model/truth by wavelength", [(d[f"spec_{ch}_{b}_ratio"], b) for b in BANDS]) for ch in SPECTRA]
                       + [(f"{ch} variance, {b}", [(d[f"spec_{ch}_{b}_model"], "model"), (d[f"spec_{ch}_{b}_truth"], "truth")])
                          for ch in SPECTRA for b in BANDS],
            "loop_current": [("LC northern extent (17 cm)", [(d.lc_north_model, "model"), (d.lc_north_truth, "truth")]),
                             ("LC northern extent error, model - truth", [(d.lc_north_error, "model - truth")]),
                             ("LCE count", [(d.lc_lce_model, "model"), (d.lc_lce_truth, "truth")]),
                             ("LC area", [(d.lc_area_model, "model"), (d.lc_area_truth, "truth")])],
            "ssh_bias": [("Gulf-mean SSH bias (cm), model - truth", [((100 * d.bias_ssh).assign_attrs(units="cm"), "model"), ((0.5 * d.lead).broadcast_like(d.bias_ssh), "+0.5 cm/day (1-4 day eval)")])],
            "energy": [("KE model/truth", [(d.ke_model / d.ke_truth, "ratio")]),
                       ("max surface speed", [(d.maxspeed_model, "model"), (d.maxspeed_truth, "truth")])]
                      + [(f"corr_change {f}", [(d[f"corr_change_{f}"], "model")]) for f in FIELDS],
        }
        for name, rows in panels.items():
            ncol = 4 if len(rows) > 5 else len(rows)
            nrow = -(-len(rows) // ncol)
            fig, axes = plt.subplots(nrow, ncol, figsize=(18, 4 * nrow), squeeze=False)
            for ax, (title, series) in zip(axes.flat, rows):
                for (da, label), color in zip(series, ("C0", "C1", "C2", "C3")):
                    _band(ax, da, label, color)
                ax.set(title=title, xlabel="lead (days)", ylabel=series[0][0].attrs.get("units", ""))
                ax.legend(fontsize=7)
            for ax in axes.flat[len(rows):]:
                ax.axis("off")
            fig.suptitle(f"{name}: median and 10-90% over starts, {span}\n{LABEL}", fontsize=9)
            fig.tight_layout()
            fig.savefig(out / f"{tag}_{name}.png", dpi=100)
            plt.close(fig)


def movie(config: Path, ckpt: Path, out: Path, starts: list[str], horizon: int, chunk: int, note: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    m = load_model(config, ckpt)
    out.mkdir(parents=True, exist_ok=True)
    todo = start_set(m.times.astype("datetime64[D]"), horizon, starts)
    names, chans = m.geo.names, ("ssh", "temp_100m")
    cols = [names.index(c) for c in chans]
    keep = [{k: np.full((s.horizon, 2, len(m.geo.region)), np.nan, np.float32) for k in ("model", "truth")} for s in todo]
    for lead, pos, step in unroll(m.module, m.window, m.times, todo, chunk):
        model, truth = step.pred[..., cols].cpu().numpy(), step.truth[..., cols].cpu().numpy()
        for i, b in enumerate(pos):
            keep[b]["model"][lead - 1], keep[b]["truth"][lead - 1] = model[i].T, truth[i].T
    static = m.meta.static.load()
    grid = lambda v: v.reshape(m.shape).T  # noqa: E731
    lon, lat = grid(static.sel(static_feature="lon").values), grid(static.sel(static_feature="lat").values)
    region = m.geo.region.cpu().numpy()
    k100 = int(np.argmin(np.abs(m.meta.level.values - 100)))
    masks = [region, region & m.meta.level_ocean.values[:, k100].astype(bool)]
    use_ffmpeg = shutil.which("ffmpeg") is not None
    for s, arr in zip(todo, keep):
        mod, tru = (np.where(np.array(masks)[None], arr[k], np.nan) for k in ("model", "truth"))
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
        meshes = []
        for r, ch in enumerate(chans):
            lo, hi = np.nanpercentile(tru[:, r], [2, 98])
            dl = float(np.nanpercentile(np.abs(mod[:, r] - tru[:, r]), 98))
            for c, (title, cmap, lim) in enumerate(((f"{ch} model", "viridis", (lo, hi)), (f"{ch} truth", "viridis", (lo, hi)),
                                                    (f"{ch} model - truth", "RdBu_r", (-dl, dl)))):
                ax = axes[r, c]
                mesh = ax.pcolormesh(lon, lat, np.full(lon.shape, np.nan), cmap=cmap, vmin=lim[0], vmax=lim[1], shading="nearest")
                fig.colorbar(mesh, ax=ax, shrink=0.8)
                ax.set_title(title, fontsize=9)
                meshes.append(mesh)
        title = fig.suptitle("", fontsize=8)

        def frame(l):
            for r in range(2):
                for c, v in enumerate((mod[l, r], tru[l, r], mod[l, r] - tru[l, r])):
                    meshes[3 * r + c].set_array(grid(v))
            title.set_text(f"{note + chr(10) if note else ''}start {s.date}  valid {s.date + (l + 1) * DAY}  lead {l + 1} d\n{LABEL}")

        stem = out / f"rollout_{s.date}"
        writer = animation.FFMpegWriter(fps=10) if use_ffmpeg else animation.PillowWriter(fps=10)
        with writer.saving(fig, f"{stem}.{'mp4' if use_ffmpeg else 'gif'}", dpi=80):
            for l in range(s.horizon):
                frame(l)
                writer.grab_frame()
        for lead in (1, 5, 10, 15, 30, 60, 90, 180, 365):
            if lead <= s.horizon:
                frame(lead - 1)
                fig.savefig(f"{stem}_lead{lead:03d}.png", dpi=80)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    dates = lambda s: s.split(",")  # noqa: E731
    for cmd in ("run", "check", "movie"):
        q = sub.add_parser(cmd)
        q.add_argument("config", type=Path)
        q.add_argument("ckpt", type=Path)
        q.add_argument("out", type=Path)
        q.add_argument("--starts", type=dates, required=cmd == "check",
                       default="2022-01-01,2022-07-01,2023-01-01".split(",") if cmd == "movie" else None)
        q.add_argument("--horizon", type=int, default=15)
        q.add_argument("--chunk", type=int, default=15)
        if cmd == "run":
            q.add_argument("--batch", type=int, default=8)
            q.add_argument("--stride-days", type=int, default=1, help="subsample the out-of-sample daily starts")
            q.add_argument("--limit", type=int, default=None, help="first n starts after ordering (smoke)")
        if cmd == "movie":
            q.add_argument("--note", default="", help="first title line of every frame, e.g. which checkpoint")
    q = sub.add_parser("figures")
    q.add_argument("stats", type=Path)
    q.add_argument("out", type=Path)
    a = p.parse_args()
    if a.cmd == "run":
        run(a.config, a.ckpt, a.out, a.horizon, a.chunk, a.batch, a.starts, a.stride_days, a.limit)
    elif a.cmd == "check":
        print(json.dumps(check(a.config, a.ckpt, a.out, a.starts, a.horizon, a.chunk)["groups"], indent=1))
    elif a.cmd == "movie":
        movie(a.config, a.ckpt, a.out, a.starts, a.horizon, a.chunk, a.note)
    else:
        figures(a.stats, a.out)


if __name__ == "__main__":
    main()
