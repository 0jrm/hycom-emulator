"""Structure score of SSH free runs: eddy census above a noise floor, persistence-diagram distance, Loop Current
separations.

    python -m hycom_emulator.structure score <fields.nc> <out_dir> [--native]
    python -m hycom_emulator.structure truth <pack_dir> <out_dir> --from 2022-01-01 --to 2024-08-31
    python -m hycom_emulator.structure floor <fields.nc> <out.json> [--n 200]
    python -m hycom_emulator.structure report <score_dir> [--title ...]

Every output carries CAVEAT: the free runs use the true future wind, sunlight and boundary band, so skill is
optimistic, and these scores inherit that until issue-time-only forcing lands.

Engine. 0-dimensional persistence of SSH on the ocean cells of a grid, 4-connected: a Kruskal pass over the edges in
descending order of min(h_i, h_j) merges components; at each merge the component with the lower peak dies (elder
rule), giving the pair (birth = its peak, death = the merge level). Superlevel pairs are the maxima (anticyclones);
the same pass on -h gives the minima (cyclones). A component that never merges (one per connected piece of the mask)
dies at the lowest value of its piece, so every pair is finite and both fields of a comparison, which share a mask,
have the same number of them. Persistence = birth - death; a feature counts when its persistence is at least the
floor.

Region. The fields' scoring region (Gulf, ocean, outside the boundary band) deeper than MIN_DEPTH_M, as
rollout_rea's Loop Current and isotach.gulf_region use: on the shelf, wind set-up and set-down (2022-09-28, Ian)
add tens of SSH extrema above the floor in a day.

Grid. Scores are computed on the 0.25 deg DUACS grid (cell centres at k/4 + 1/8): the mean of the emulator cells
whose centres fall in each cell, a cell kept when at least half of its emulator cells are in the region. This is the
grid the floor was calibrated on (persistence-ocean, DUACS 2017-2021), so the floor transfers without a new null
model; `floor` reruns the correlated-noise null on any grid to show what it would be (--native scores on the
emulator's own grid with that recalibrated floor).

Loop Current. SSH minus its area-weighted (cos lat) mean over the region. The seed is the highest of the region's
0.25 deg cells in YUCATAN_BOX. For every cell the same Kruskal pass records the level at which its component joins
the seed's (its neck saddle) and the seed side's peak at that moment; a cell's neck depth is
min(h, seed-side peak) - saddle, zero for cells on the seed's own branch. The LC is the 17 cm component holding the
seed. A detachment is a 17 cm component of at least LCE_MIN_KM2 that overlaps the previous day's LC. It is tracked by
overlap; while it is back inside the LC its rejoin margin is saddle - 17 cm at its highest cell. A rejoin counts when
the margin is at least the floor for more than REJOIN_DAYS days in a row; a detachment is a separation when no
counted rejoin follows within SEPARATION_DAYS (persistence-ocean's rule as briefed; the reference code is not yet in
hand, see PENDING).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

CAVEAT = ("free runs with the true future wind, sunlight and boundary band: skill is optimistic, and structure scores "
          "inherit that until issue-time-only forcing lands")
PENDING = ("engine written from the brief, not yet checked against persistence-ocean's cubical_h0.py, merge_tree.py, "
           "loop_current.py and null_model.py (4-connectivity, essential-class death, rejoin margin and floor "
           "quantile are this module's readings)")
QUARTER_DEG = 0.25
EDDY_FLOOR_M = 0.055  # per eddy, open Gulf, 0.25 deg DUACS (persistence-ocean; 5-6 cm)
LC_FLOOR_M = 0.043  # per feature in the Loop Current windows, 0.25 deg DUACS (persistence-ocean)
NOISE_M, NOISE_KM = 0.035, 50.0  # DUACS mapping error against SWOT KaRIn (3.3-3.8 cm, 50 km, 8 days)
LC_LEVEL_M = 0.17
MIN_DEPTH_M = 500.0  # scored water: storm set-up and set-down on the shelf make SSH extrema that are not eddies
LCE_MIN_KM2 = 1000.0  # rollout_rea.LCE_MIN_KM2
YUCATAN_BOX = {"lon": (-86.6, -85.2), "lat": (21.5, 22.0)}
REJOIN_DAYS = 8  # a rejoin counts when its margin holds for more than this many days in a row
SEPARATION_DAYS = 30  # a detachment is a separation when no counted rejoin follows within this many days
EARTH_KM_PER_DEG = 111.32


# ---------------------------------------------------------------------------------------------------- engine

@dataclass(frozen=True)
class Graph:
    """Ocean cells of a (ny, nx) grid and their 4-neighbour edges, as flat indices into the cells."""
    shape: tuple[int, int]
    cells: np.ndarray  # (n,) flat (ny * nx) index of each ocean cell
    edges: np.ndarray  # (m, 2) cell numbers


def graph(mask: np.ndarray) -> Graph:
    number = np.full(mask.shape, -1)
    number[mask] = np.arange(mask.sum())
    pairs = [np.stack([number[:, :-1][mask[:, :-1] & mask[:, 1:]], number[:, 1:][mask[:, :-1] & mask[:, 1:]]], 1),
             np.stack([number[:-1][mask[:-1] & mask[1:]], number[1:][mask[:-1] & mask[1:]]], 1)]
    return Graph(mask.shape, np.flatnonzero(mask), np.concatenate(pairs).astype(np.int64))


@dataclass
class Merge:
    """One superlevel pass. pairs: (k, 2) (birth, death) with birth > death, incl. one pair per mask piece;
    peak: (k,) cell number of each pair's maximum. With a seed: saddle and seed_peak per cell (saddle = +inf on the
    seed's own branch)."""
    pairs: np.ndarray
    peak: np.ndarray
    essential: np.ndarray  # (k,) bool: the pair of a mask piece's highest cell, dying at the piece's lowest value
    saddle: np.ndarray | None = None
    seed_peak: np.ndarray | None = None

    def neck_depth(self, h: np.ndarray) -> np.ndarray:
        """(n,) min(h, seed-side peak) - saddle, 0 on the seed's branch."""
        d = np.minimum(h, self.seed_peak) - self.saddle
        return np.where(np.isfinite(self.saddle), np.maximum(d, 0.0), 0.0)


def superlevel(h: np.ndarray, g: Graph, seed: int | None = None) -> Merge:
    """h: (n,) values on g's cells, all finite. Ties break by cell number, so the result is deterministic."""
    n = len(h)
    rank = np.empty(n, np.int64)
    rank[np.lexsort((np.arange(n), -h))] = np.arange(n)  # 0 = highest
    a, b = g.edges[:, 0], g.edges[:, 1]
    low = np.where(rank[a] > rank[b], a, b)  # the edge enters at its lower end
    order = np.argsort(rank[low], kind="stable")
    parent = np.arange(n)
    peak = np.arange(n)  # per root: its highest cell
    lowest = h.copy()  # per root: its lowest value
    members = [[i] for i in range(n)] if seed is not None else None
    saddle = np.full(n, np.inf) if seed is not None else None
    seed_peak = np.full(n, np.nan) if seed is not None else None
    births, deaths, peaks = [], [], []

    def find(i):
        r = i
        while parent[r] != r:
            r = parent[r]
        while parent[i] != r:
            parent[i], i = r, parent[i]
        return r

    for e in order:
        ra, rb = find(a[e]), find(b[e])
        if ra == rb:
            continue
        level = h[low[e]]
        if rank[peak[ra]] > rank[peak[rb]]:
            ra, rb = rb, ra  # ra: the elder (higher peak)
        if h[peak[rb]] > level:
            births.append(h[peak[rb]]), deaths.append(level), peaks.append(peak[rb])
        if seed is not None:
            sa, sb = find(seed) == ra, find(seed) == rb
            if sa or sb:
                young, old = (rb, ra) if sa else (ra, rb)
                idx = np.array(members[young])
                saddle[idx], seed_peak[idx] = level, h[peak[old]]
            big, small = (ra, rb) if len(members[ra]) >= len(members[rb]) else (rb, ra)
            members[big].extend(members[small])
            members[small] = []
            parent[small] = big
            root = big
        else:
            parent[rb] = ra
            root = ra
        peak[root] = peak[ra]
        lowest[root] = min(lowest[ra], lowest[rb])
    finite = len(births)
    for r in np.unique([find(i) for i in range(n)]):
        births.append(h[peak[r]]), deaths.append(lowest[r]), peaks.append(peak[r])
    pairs = np.array([births, deaths], dtype=float).T.reshape(-1, 2)
    return Merge(pairs, np.array(peaks, dtype=np.int64), np.arange(len(births)) >= finite, saddle, seed_peak)


def diagrams(h: np.ndarray, g: Graph) -> dict[str, np.ndarray]:
    """{"max": pairs of maxima in m, "min": pairs of minima as (-birth, -death) of -h, so birth > death in both}."""
    return {"max": superlevel(h, g).pairs, "min": superlevel(-h, g).pairs}


def persistence(pairs: np.ndarray) -> np.ndarray:
    return pairs[:, 0] - pairs[:, 1]


def above(pairs: np.ndarray, floor: float) -> np.ndarray:
    return pairs[persistence(pairs) >= floor]


# ------------------------------------------------------------------------------------------------- distances

def _costs(d1: np.ndarray, d2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """L-infinity point costs and each point's cost to the diagonal."""
    c = np.maximum(np.abs(d1[:, None, 0] - d2[None, :, 0]), np.abs(d1[:, None, 1] - d2[None, :, 1]))
    return c, persistence(d1) / 2, persistence(d2) / 2


def _augmented(d1: np.ndarray, d2: np.ndarray, big: float) -> np.ndarray:
    n, m = len(d1), len(d2)
    c, g1, g2 = _costs(d1, d2)
    top = np.full((n, n), big)
    np.fill_diagonal(top, g1)
    bot = np.full((m, m), big)
    np.fill_diagonal(bot, g2)
    return np.block([[c, top], [bot, np.zeros((m, n))]])


def wasserstein(d1: np.ndarray, d2: np.ndarray, p: float = 1.0) -> float:
    """p-Wasserstein distance between diagrams (L-infinity ground metric, unmatched points go to the diagonal)."""
    if len(d1) + len(d2) == 0:
        return 0.0
    m = _augmented(d1, d2, np.inf)
    big = 1e3 * (np.nanmax(np.where(np.isfinite(m), m, np.nan)) + 1.0)
    m = np.where(np.isfinite(m), m, big) ** p
    r, c = linear_sum_assignment(m)
    return float(m[r, c].sum() ** (1 / p))


def bottleneck(d1: np.ndarray, d2: np.ndarray) -> float:
    """Bottleneck distance: the smallest t such that a matching uses no cost above t."""
    if len(d1) + len(d2) == 0:
        return 0.0
    m = _augmented(d1, d2, np.inf)
    cand = np.unique(m[np.isfinite(m)])
    lo, hi = 0, len(cand) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        ok = maximum_bipartite_matching(csr_matrix(m <= cand[mid]), perm_type="column")
        lo, hi = (lo, mid) if (ok >= 0).all() else (mid + 1, hi)
    return float(cand[lo])


# ---------------------------------------------------------------------------------------------------- grids

@dataclass(frozen=True)
class Grid:
    """A (ny, nx) scoring grid: lon, lat (ny, nx), the region mask, and its Graph."""
    lon: np.ndarray
    lat: np.ndarray
    mask: np.ndarray
    g: Graph
    cell_km2: np.ndarray  # (ny, nx)

    def weights(self) -> np.ndarray:
        return np.cos(np.deg2rad(self.lat))[self.mask]


@dataclass(frozen=True)
class Regrid:
    """Fine flat cells -> a coarse Grid by the mean over the fine region cells in each coarse cell."""
    grid: Grid
    average: csr_matrix  # (coarse cells, fine grid) rows summing to 1

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """(..., n_fine) -> (..., coarse cells)."""
        return (self.average @ x.reshape(-1, x.shape[-1]).T).T.reshape(*x.shape[:-1], -1)


def quarter(lon: np.ndarray, lat: np.ndarray, region: np.ndarray, deg: float = QUARTER_DEG) -> Regrid:
    """lon, lat, region: flat fine-grid arrays. Coarse cell centres at k * deg + deg / 2 (the DUACS 0.25 deg grid)."""
    ix, iy = np.floor(lon / deg).astype(int), np.floor(lat / deg).astype(int)
    x0, y0 = ix.min(), iy.min()
    nx, ny = ix.max() - x0 + 1, iy.max() - y0 + 1
    cell = (iy - y0) * nx + (ix - x0)
    total = np.bincount(cell, minlength=nx * ny)
    inside = np.bincount(cell[region], minlength=nx * ny)
    keep = (inside > 0) & (2 * inside >= total)
    mask = keep.reshape(ny, nx)
    g = graph(mask)
    number = np.full(nx * ny, -1)
    number[g.cells] = np.arange(len(g.cells))
    use = np.flatnonzero(region & keep[cell])
    coarse = number[cell[use]]
    count = np.bincount(coarse, minlength=len(g.cells))
    average = csr_matrix((1.0 / count[coarse], (coarse, use)), shape=(len(g.cells), len(lon)))
    clon, clat = np.meshgrid((np.arange(nx) + x0 + 0.5) * deg, (np.arange(ny) + y0 + 0.5) * deg)
    km2 = (EARTH_KM_PER_DEG * deg) ** 2 * np.cos(np.deg2rad(clat))
    grid = Grid(clon, clat, mask, g, km2)
    return Regrid(grid, average)


def native(lon2: np.ndarray, lat2: np.ndarray, mask: np.ndarray, dlon: float) -> Grid:
    """The emulator's own (ny, nx) grid as a scoring Grid."""
    km2 = (EARTH_KM_PER_DEG * dlon) ** 2 * np.cos(np.deg2rad(lat2))
    return Grid(lon2, lat2, mask, graph(mask), km2)


def demean(h: np.ndarray, grid: Grid) -> np.ndarray:
    w = grid.weights()
    return h - (h * w).sum(-1, keepdims=True) / w.sum()


# ----------------------------------------------------------------------------------------------- noise floor

def correlated_noise(grid: Grid, n: int, sigma_m: float = NOISE_M, corr_km: float = NOISE_KM, seed: int = 0) -> np.ndarray:
    """(n, cells) Gaussian noise of std sigma_m with covariance exp(-r^2 / (2 corr_km^2)) on the grid's rectangle
    (a white field smoothed by a Gaussian of std corr_km / sqrt(2), then rescaled on the region)."""
    rng = np.random.default_rng(seed)
    ny, nx = grid.mask.shape
    dy_km = EARTH_KM_PER_DEG * float(np.median(np.diff(grid.lat[:, 0])))
    dx_km = EARTH_KM_PER_DEG * float(np.median(np.diff(grid.lon[0]))) * float(np.cos(np.deg2rad(grid.lat[grid.mask].mean())))
    s = corr_km / np.sqrt(2)
    pad = int(np.ceil(4 * s / min(dx_km, dy_km)))
    out = np.empty((n, grid.mask.sum()))
    for i in range(n):
        f = ndimage.gaussian_filter(rng.standard_normal((ny + 2 * pad, nx + 2 * pad)), (s / dy_km, s / dx_km), mode="wrap")
        v = f[pad:pad + ny, pad:pad + nx][grid.mask]
        out[i] = v / v.std() * sigma_m
    return out


def noise_floor(grid: Grid, n: int = 200, q: float = 0.95, **kw) -> dict:
    """Persistence of the features of pure correlated noise on the grid, maxima and minima pooled (the essential
    pairs left out): the q quantile is the floor below which a feature is indistinguishable from mapping error."""
    noise = correlated_noise(grid, n, **kw)
    p = np.concatenate([persistence(m.pairs[~m.essential]) for x in noise for m in (superlevel(x, grid.g), superlevel(-x, grid.g))])
    return {"floor_m": float(np.quantile(p, q)), "q": q, "n_fields": n, "features_per_field": len(p) / n,
            "median_m": float(np.median(p)), "p99_m": float(np.quantile(p, 0.99)), "cells": int(grid.mask.sum()),
            "sigma_m": kw.get("sigma_m", NOISE_M), "corr_km": kw.get("corr_km", NOISE_KM)}


# ---------------------------------------------------------------------------------------------- Loop Current

def yucatan_cells(grid: Grid) -> np.ndarray:
    """Cell numbers of the region inside YUCATAN_BOX (centres)."""
    lon, lat = grid.lon[grid.mask], grid.lat[grid.mask]
    box = ((lon >= YUCATAN_BOX["lon"][0]) & (lon <= YUCATAN_BOX["lon"][1])
           & (lat >= YUCATAN_BOX["lat"][0]) & (lat <= YUCATAN_BOX["lat"][1]))
    if not box.any():
        raise ValueError("no region cell in the Yucatan box")
    return np.flatnonzero(box)


@dataclass
class LCDay:
    """One day of demeaned SSH h (cells): the merge with the seed, the LC (bool cells) and the other 17 cm
    components of at least LCE_MIN_KM2 (list of cell-number arrays)."""
    h: np.ndarray
    merge: Merge
    lc: np.ndarray
    detached: list[np.ndarray]

    @property
    def neck(self) -> np.ndarray:
        return self.merge.neck_depth(self.h)


def lc_day(h: np.ndarray, grid: Grid, box: np.ndarray) -> LCDay:
    seed = int(box[np.argmax(h[box])])
    m = superlevel(h, grid.g, seed)
    img = np.zeros(grid.mask.shape, bool)
    img[grid.mask] = h >= LC_LEVEL_M
    lab, _ = ndimage.label(img)  # 4-connected, as the engine
    lab = lab[grid.mask]
    km2 = grid.cell_km2[grid.mask]
    lc = (lab == lab[seed]) & (lab[seed] > 0)
    det = []
    for k in np.unique(lab[(lab > 0) & ~lc]):
        cells = np.flatnonzero(lab == k)
        if km2[cells].sum() >= LCE_MIN_KM2:
            det.append(cells)
    return LCDay(h, m, lc, det)


@dataclass
class Event:
    detached: int  # day index of the detachment
    footprint: np.ndarray  # cells at detachment
    status: str = "open"  # open, separation, rejoined, unresolved (series ended before SEPARATION_DAYS)
    rejoined: int | None = None  # first day of the counted rejoin run
    neck: dict = field(default_factory=dict)  # day index -> neck depth of the tracked feature (m)
    margin: dict = field(default_factory=dict)  # day index -> rejoin margin while inside the LC (m)


def _dilate(cells: np.ndarray, grid: Grid, k: int = 1) -> np.ndarray:
    img = np.zeros(grid.mask.shape, bool)
    img.flat[grid.g.cells[cells]] = True
    img = ndimage.binary_dilation(img, iterations=k) & grid.mask
    return np.flatnonzero(img[grid.mask])


def separations(days: list[LCDay], grid: Grid, floor: float = LC_FLOOR_M, before: int = 20) -> list[Event]:
    """Detachments of the series and their fate, with the tracked feature's neck depth from `before` days ahead of
    each detachment (on its footprint dilated by two cells) to the end of tracking. Day 0 only seeds the LC."""
    track_days = SEPARATION_DAYS + REJOIN_DAYS + 1  # a rejoin starting on day 30 is known on day 39
    events: list[Event] = []
    active: list[list] = []  # [event, footprint, days in a row with margin >= floor]
    for t in range(1, len(days)):
        d, prev = days[t], days[t - 1]
        neck = d.neck
        for a in active:
            ev, foot, run = a
            near = _dilate(foot, grid)
            hit = [c for c in d.detached if np.isin(c, near).any()]
            if hit:
                a[1], a[2] = np.unique(np.concatenate(hit)), 0
                ev.neck[t] = float(neck[a[1]].max())
            elif d.lc[near].any():
                cells = near[d.lc[near]]
                top = cells[np.argmax(d.h[cells])]
                ev.margin[t] = float(d.merge.saddle[top] - LC_LEVEL_M)  # +inf on the seed's own branch
                ev.neck[t] = float(neck[cells].max())
                a[2] = run + 1 if ev.margin[t] >= floor else 0
                if a[2] > REJOIN_DAYS and ev.rejoined is None:
                    ev.rejoined = t - a[2] + 1
            else:
                ev.neck[t] = 0.0  # decayed below 17 cm
        active = [a for a in active if a[0].rejoined is None and t - a[0].detached < track_days]
        tracked = np.concatenate([_dilate(a[1], grid) for a in active]) if active else np.array([], int)
        for c in d.detached:
            if prev.lc[c].any() and not np.isin(c, tracked).any():
                ev = Event(t, c)
                wide = _dilate(c, grid, 2)
                ev.neck = {s: float(days[s].neck[wide].max()) for s in range(max(0, t - before), t)} | {t: float(neck[c].max())}
                events.append(ev)
                active.append([ev, c, 0])
    last = len(days) - 1
    for ev in events:
        if ev.rejoined is not None and ev.rejoined - ev.detached <= SEPARATION_DAYS:
            ev.status = "rejoined"
        elif last - ev.detached >= track_days or (ev.rejoined is None and last - ev.detached >= SEPARATION_DAYS
                                                   and _no_run_possible(ev, last, floor)):
            ev.status = "separation"
        else:
            ev.status = "unresolved"
    return events


def event_sizes(ev: Event, days: list[LCDay], grid: Grid) -> dict:
    """Areas (km2) of the detached piece, the LC the day before and the day of detachment, and the piece's peak (m)."""
    km2 = grid.cell_km2[grid.mask]
    return {"ring_km2": float(km2[ev.footprint].sum()), "lc_before_km2": float(km2[days[ev.detached - 1].lc].sum()),
            "lc_after_km2": float(km2[days[ev.detached].lc].sum()), "ring_peak_m": float(days[ev.detached].h[ev.footprint].max())}


def _no_run_possible(ev: Event, last: int, floor: float) -> bool:
    """A rejoin run still open at the series end could reach REJOIN_DAYS + 1 only if it started by day 30."""
    run = 0
    for t in range(ev.detached + 1, last + 1):
        run = run + 1 if ev.margin.get(t, -np.inf) >= floor else 0
    return last - run + 1 > ev.detached + SEPARATION_DAYS


# --------------------------------------------------------------------------------------------------- scoring

def census(pairs: np.ndarray, floor: float) -> dict:
    p = persistence(pairs)
    p = p[p >= floor]
    return {"count": len(p), "amp_median": float(np.median(p)) if len(p) else np.nan,
            "amp_p90": float(np.quantile(p, 0.9)) if len(p) else np.nan, "amp_sum": float(p.sum())}


CENSUS = ("count", "amp_median", "amp_p90", "amp_sum")


def score_pair(model: np.ndarray, truth: np.ndarray, start: np.ndarray, g: Graph, floor: float) -> dict:
    """Census of model and truth, and diagram distances model-truth and persistence(start)-truth, per kind."""
    dm, dt, ds = diagrams(model, g), diagrams(truth, g), diagrams(start, g)
    out = {}
    for k in ("max", "min"):
        am, at, as_ = (above(d[k], floor) for d in (dm, dt, ds))
        for who, pairs in (("model", dm[k]), ("truth", dt[k])):
            out |= {f"{k}_{c}_{who}": v for c, v in census(pairs, floor).items()}
        out[f"{k}_w1_model"], out[f"{k}_w1_pers"] = wasserstein(am, at), wasserstein(as_, at)
        out[f"{k}_bottleneck_model"], out[f"{k}_bottleneck_pers"] = bottleneck(am, at), bottleneck(as_, at)
    return out


def open_fields(path: Path) -> xr.Dataset:
    return xr.open_dataset(path)


def scoring_grid(ds: xr.Dataset, use_native: bool) -> tuple[Grid, callable]:
    """The Grid and a map from flat fine fields (..., grid_index) to its cells."""
    lon, lat = ds.lon.values, ds.lat.values
    region = ds.region.values.astype(bool) & (np.nan_to_num(ds.depth.values) > MIN_DEPTH_M)
    if not use_native:
        r = quarter(lon, lat, region)
        return r.grid, r
    nx, ny = int(ds.attrs["nx"]), int(ds.attrs["ny"])
    img = lambda v: v.reshape(nx, ny).T  # noqa: E731
    mask = img(region)
    grid = native(img(lon), img(lat), mask, float(ds.attrs["dlon"]))
    flat = np.flatnonzero(region.reshape(nx, ny).T.ravel())  # image order
    perm = np.arange(nx * ny).reshape(nx, ny).T.ravel()[flat]
    return grid, lambda x: x[..., perm]


def score(fields: Path, out: Path, use_native: bool = False, floor: float | None = None, lc_floor: float | None = None) -> xr.Dataset:
    ds = open_fields(fields)
    grid, to = scoring_grid(ds, use_native)
    if floor is None:
        floor = EDDY_FLOOR_M if not use_native else noise_floor(grid, 100)["floor_m"]
    lc_floor = LC_FLOOR_M if lc_floor is None else lc_floor
    box = yucatan_cells(grid)
    rows, events = [], []
    for i, s in enumerate(ds.start.values):
        h = int(ds.horizon.values[i])
        x0 = demean(to(ds.ssh_start.values[i].astype(float)), grid)
        model = demean(to(ds.ssh_model.values[i, :h].astype(float)), grid)
        truth = demean(to(ds.ssh_truth.values[i, :h].astype(float)), grid)
        rows.append([score_pair(model[l], truth[l], x0, grid.g, floor) for l in range(h)])
        for who, seq in (("model", model), ("truth", truth)):
            days = [lc_day(x, grid, box) for x in (x0, *seq)]
            for ev in separations(days, grid, lc_floor):
                events.append({"start": str(s)[:10], "who": who, "detached_lead": ev.detached,
                               "detached": str(np.datetime64(str(s)[:10]) + np.timedelta64(ev.detached, "D")), "status": ev.status,
                               "rejoined_lead": ev.rejoined, **event_sizes(ev, days, grid),
                               "neck": {int(k) - ev.detached: v for k, v in sorted(ev.neck.items())}})
    keys = list(rows[0][0])
    lead = np.arange(1, int(ds.horizon.max()) + 1)
    data = {k: (("start", "lead"), np.array([[r[l][k] if l < len(r) else np.nan for l in range(len(lead))] for r in rows]))
            for k in keys}
    res = xr.Dataset(data, {"start": ds.start.values, "lead": lead},
                     {"caveat": CAVEAT, "pending": PENDING, "grid": "native" if use_native else "0.25 deg DUACS cells",
                      "eddy_floor_m": floor, "lc_floor_m": lc_floor, "cells": int(grid.mask.sum()),
                      "fields": str(fields), **{f"fields_{k}": str(v) for k, v in ds.attrs.items() if k in ("checkpoint", "checkpoint_md5", "code_sha")}})
    out.mkdir(parents=True, exist_ok=True)
    res.to_netcdf(out / "structure.nc")
    (out / "events.json").write_text(json.dumps({"caveat": CAVEAT, "pending": PENDING, "events": events}, indent=1))
    return res


def truth_series(pack: Path, out: Path, t_from: str, t_to: str, floor: float = LC_FLOOR_M) -> list[dict]:
    """Loop Current separations of the pack's own SSH (the reanalysis truth), day by day over [t_from, t_to]."""
    meta = xr.open_zarr(pack / "meta.zarr", consolidated=True)
    names = [str(n) for n in meta.state_feature.values]
    st = lambda f: meta.static.sel(static_feature=f).values  # noqa: E731
    region = (st("gulf").astype(bool) & st("ocean").astype(bool) & ~meta.boundary_mask.values.astype(bool)
              & (np.nan_to_num(st("depth")) > MIN_DEPTH_M))
    r = quarter(st("lon"), st("lat"), region)
    times = meta.time.values.astype("datetime64[D]")
    rows = np.flatnonzero((times >= np.datetime64(t_from)) & (times <= np.datetime64(t_to)))
    state = np.load(pack / "state.npy", mmap_mode="r")
    j = names.index("ssh")
    if state.dtype != np.float32:
        raise ValueError(f"{pack}/state.npy is {state.dtype}; truth reads a float32 pack")
    ssh = np.stack([np.asarray(state[t, :, j], dtype=float) for t in rows])
    h = demean(r(ssh), r.grid)
    box = yucatan_cells(r.grid)
    days = [lc_day(x, r.grid, box) for x in h]
    evs = separations(days, r.grid, floor)
    res = [{"detached": str(times[rows[ev.detached]]), "status": ev.status,
            "rejoined": str(times[rows[ev.rejoined]]) if ev.rejoined is not None else None, **event_sizes(ev, days, r.grid),
            "neck": {int(k) - ev.detached: v for k, v in sorted(ev.neck.items())}} for ev in evs]
    out.mkdir(parents=True, exist_ok=True)
    census_rows = [{k: census(d, EDDY_FLOOR_M)["count"] for k, d in diagrams(x, r.grid.g).items()} for x in h]
    (out / "truth_events.json").write_text(json.dumps({"caveat": "reanalysis truth, no model", "pending": PENDING,
                                                       "from": t_from, "to": t_to, "lc_floor_m": floor, "events": res}, indent=1))
    xr.Dataset({"max_count": ("time", [c["max"] for c in census_rows]), "min_count": ("time", [c["min"] for c in census_rows]),
                "neck_lc_max": ("time", [float(d.neck.max()) for d in days]),
                "lc_area_km2": ("time", [float(r.grid.cell_km2[r.grid.mask][d.lc].sum()) for d in days])},
               {"time": times[rows].astype("datetime64[ns]")}, {"eddy_floor_m": EDDY_FLOOR_M}).to_netcdf(out / "truth_series.nc")
    return res


MATCH_DAYS = 15  # a model separation within this many days of a truth one is the same event


def match_events(events: list[dict]) -> list[dict]:
    """Per start, each truth separation and the nearest model separation within MATCH_DAYS (or none)."""
    rows = []
    for start in sorted({e["start"] for e in events}):
        mine = [e for e in events if e["start"] == start and e["status"] == "separation"]
        truth, model = [e for e in mine if e["who"] == "truth"], [e for e in mine if e["who"] == "model"]
        used = set()
        for t in truth:
            near = sorted((abs(m["detached_lead"] - t["detached_lead"]), i) for i, m in enumerate(model) if i not in used)
            m = model[near[0][1]] if near and near[0][0] <= MATCH_DAYS else None
            if m is not None:
                used.add(near[0][1])
            rows.append({"start": start, "truth_lead": t["detached_lead"], "model_lead": m and m["detached_lead"],
                         "error_days": m and m["detached_lead"] - t["detached_lead"],
                         "truth_neck0": t["neck"]["0"], "model_neck0": m and m["neck"]["0"],
                         "truth_ring_km2": t["ring_km2"], "model_ring_km2": m and m["ring_km2"]})
        rows += [{"start": start, "truth_lead": None, "model_lead": m["detached_lead"], "error_days": None,
                  "truth_neck0": None, "model_neck0": m["neck"]["0"], "truth_ring_km2": None, "model_ring_km2": m["ring_km2"]}
                 for i, m in enumerate(model) if i not in used]
    return rows


def report(score_dir: Path, title: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = xr.open_dataset(score_dir / "structure.nc")
    events = json.loads((score_dir / "events.json").read_text())["events"]
    lead = res.lead.values
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for r, k in enumerate(("max", "min")):
        what = "maxima (anticyclones, LC)" if k == "max" else "minima (cyclones)"
        for who, c in (("model", "C0"), ("truth", "C1")):
            ax[r, 0].plot(lead, res[f"{k}_count_{who}"].mean("start"), c, label=who)
            ax[r, 1].plot(lead, res[f"{k}_amp_sum_{who}"].mean("start") * 100, c, label=who)
        ax[r, 2].plot(lead, res[f"{k}_w1_model"].mean("start") * 100, "C0", label="model vs truth")
        ax[r, 2].plot(lead, res[f"{k}_w1_pers"].mean("start") * 100, "C2", label="persistence vs truth")
        ax[r, 0].set(title=f"{what}: count above {res.eddy_floor_m * 100:.1f} cm", xlabel="lead (days)")
        ax[r, 1].set(title=f"{what}: summed persistence (cm)", xlabel="lead (days)")
        ax[r, 2].set(title=f"{what}: W1 diagram distance (cm)", xlabel="lead (days)")
        for a in ax[r]:
            a.legend(fontsize=8)
    fig.suptitle(f"{title}\n{res.grid}, mean over {res.sizes['start']} starts. {CAVEAT}", fontsize=8)
    fig.tight_layout()
    fig.savefig(score_dir / "structure.png", dpi=90)
    plt.close(fig)
    rows = match_events(events)
    ok = [r for r in rows if r["error_days"] is not None]
    pick = lambda leads: {l: lead.tolist().index(l) for l in leads if l in lead}  # noqa: E731
    at = pick((1, 7, 14, 21, 30, 45, 60))
    tab = ["| lead (d) | " + " | ".join(map(str, at)) + " |", "|---" * (len(at) + 1) + "|"]
    for k in ("max", "min"):
        for name, v in ((f"{k} count model / truth", res[f"{k}_count_model"].mean("start") / res[f"{k}_count_truth"].mean("start")),
                        (f"{k} summed persistence model / truth", res[f"{k}_amp_sum_model"].mean("start") / res[f"{k}_amp_sum_truth"].mean("start")),
                        (f"{k} W1 model / persistence", res[f"{k}_w1_model"].mean("start") / res[f"{k}_w1_pers"].mean("start"))):
            tab.append(f"| {name} | " + " | ".join(f"{float(v[i]):.2f}" for i in at.values()) + " |")
    ev = ["| start | truth sep. lead | model sep. lead | error (d) | neck at detachment truth / model (cm) | ring area truth / model (1000 km2) |",
          "|---|---|---|---|---|---|"]
    f = lambda x, s=1: "-" if x is None else f"{x * s:.0f}"  # noqa: E731
    for r in rows:
        ev.append(f"| {r['start']} | {f(r['truth_lead'])} | {f(r['model_lead'])} | {f(r['error_days'])} | "
                  f"{f(r['truth_neck0'], 100)} / {f(r['model_neck0'], 100)} | {f(r['truth_ring_km2'], 1e-3)} / {f(r['model_ring_km2'], 1e-3)} |")
    n_t = sum(r["truth_lead"] is not None for r in rows)
    n_m = sum(r["model_lead"] is not None for r in rows)
    err = np.array([r["error_days"] for r in ok], float)
    md = "\n".join([f"# {title}", "", f"**Caveat.** {CAVEAT}.", "", f"**Pending.** {PENDING}.", "",
                     f"Grid: {res.grid} ({res.cells} cells); eddy floor {res.eddy_floor_m * 100:.1f} cm, LC floor {res.lc_floor_m * 100:.1f} cm; "
                     f"{res.sizes['start']} starts, leads 1-{lead[-1]}. Ratios are of means over starts.", "", *tab, "",
                     f"Separations inside the free runs (confirmed: no counted rejoin within {SEPARATION_DAYS} days, so only "
                     f"detachments by lead {lead[-1] - SEPARATION_DAYS - REJOIN_DAYS - 1} are always decidable): truth {n_t}, model {n_m}, "
                     f"matched within {MATCH_DAYS} days {len(ok)}" + (f", median |error| {np.median(np.abs(err)):.0f} d, mean error {err.mean():+.1f} d." if len(ok) else "."),
                     "", *ev, "", "![structure](structure.png)", ""])
    (score_dir / "report.md").write_text(md)
    return md


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("score")
    q.add_argument("fields", type=Path)
    q.add_argument("out", type=Path)
    q.add_argument("--native", action="store_true", help="score on the emulator grid with a recalibrated floor")
    q = sub.add_parser("truth")
    q.add_argument("pack", type=Path)
    q.add_argument("out", type=Path)
    q.add_argument("--from", dest="t_from", required=True)
    q.add_argument("--to", dest="t_to", required=True)
    q = sub.add_parser("report")
    q.add_argument("score_dir", type=Path)
    q.add_argument("--title", default="SSH structure score")
    q = sub.add_parser("floor")
    q.add_argument("fields", type=Path)
    q.add_argument("out", type=Path)
    q.add_argument("--n", type=int, default=200)
    a = p.parse_args()
    if a.cmd == "score":
        score(a.fields, a.out, a.native)
    elif a.cmd == "report":
        print(report(a.score_dir, a.title))
    elif a.cmd == "truth":
        print(json.dumps(truth_series(a.pack, a.out, a.t_from, a.t_to), indent=1))
    else:
        ds = open_fields(a.fields)
        res = {name: noise_floor(scoring_grid(ds, nat)[0], a.n) for name, nat in (("quarter", False), ("native", True))}
        a.out.write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
