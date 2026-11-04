"""Structure score of SSH free runs: eddy census above a noise floor, persistence-diagram distance, Loop Current
separations and neck depth.

    python -m hycom_emulator.structure score <fields.nc> <out_dir> [--native]
    python -m hycom_emulator.structure truth <pack_dir> <out_dir> --from 2022-01-01 --to 2024-08-31
    python -m hycom_emulator.structure floor <fields.nc> <out.json> [--n 200]
    python -m hycom_emulator.structure report <score_dir> [--title ...]

Every output carries CAVEAT: the free runs use the true future wind, sunlight and boundary band, so skill is
optimistic, and these scores inherit that until issue-time-only forcing lands.

The engine and the Loop Current rules are hycom_emulator.topology, vendored from persistence-ocean (205e3c3), so the
numbers mean what they mean there:

- Eddy census: H0 persistence of SSH with 8-connectivity and an open boundary (a component dies when it reaches land
  or the grid edge), maxima = anticyclones, minima = cyclones, as persistence-ocean's atlas audit and its
  signal-plus-noise floor use. A feature counts when its persistence is at least EDDY_FLOOR_M.
- Diagram distance: W1 and bottleneck (L-infinity ground metric) between the census diagrams of model and truth,
  pairs above the floor only, and the same from the start state (persistence) to the truth.
- Loop Current: SSH minus its plain Gulf mean on the 0.25 deg Gulf (topology.gulf_grid: the ocean piece inside the
  three Gulf cuts that holds the Yucatan seed box, ten cells). LC17 detachments, each piece followed to its fate;
  a rejoin counts when its touch depth (connection level to the seed minus 17 cm) is at least LC_FLOOR_M on more
  than 8 days in a row; a final separation is a piece north of 24 N or west of 87 W that stays off LC17 more than
  30 days without a counted rejoin. Each event's neck depth, min(h_max, seed-side peak) - h_saddle, is followed
  from the separation back to the start (or 180 days) and 30 days forward.

Grid. Scores are computed on the 0.25 deg DUACS cells (centres at k/4 + 1/8), the mean of the emulator cells in each,
a cell kept when at least half of its emulator cells are in the region: the grid every floor and rule was set on.
`--native` scores the census on the emulator grid with the pure-noise floor recalibrated there (`floor`), as a
sensitivity only (the signal-plus-noise floor that the 0.25 deg score uses has no native counterpart).

Region. The fields' region (Gulf, ocean, outside the boundary band). The census leaves out water shallower than
MIN_DEPTH_M: in daily-mean HYCOM SSH, storm set-up and set-down on the shelf (2022-09-28, Ian) add up to ten
extrema above the floor in one day, which mapped DUACS does not show. The Loop Current keeps the shelf, as the
reference does.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from hycom_emulator import topology as tp

CAVEAT = ("free runs with the true future wind, sunlight and boundary band: skill is optimistic, and structure scores "
          "inherit that until issue-time-only forcing lands")
REFERENCE = "engine and LC rules vendored from persistence-ocean feat/write-up-assets 205e3c3 (hycom_emulator.topology)"
QUARTER_DEG = 0.25
EDDY_FLOOR_M = 0.055  # persistence-ocean signal-plus-noise floor per eddy, open Gulf, KaRIn central (5-6 cm)
LC_FLOOR_M = 0.04324  # persistence-ocean touch rule min depth: per-feature q95, L = 50 km, snp central, LCE windows
TOUCH_DAYS = 8  # ERROR_TIME_SCALE_DAYS: a rejoin counts when deep enough on more than this many days in a row
NOISE_M, NOISE_KM = 0.035, 50.0  # DUACS mapping error against SWOT KaRIn (3.3-3.8 cm), Gaussian L of the error
MIN_DEPTH_M = 500.0
BACK_TRACK_MAX_DAYS = 180
CENSUS = {"connectivity": 8, "open_boundary": True}
EARTH_KM_PER_DEG = 111.32
POLARITY = {"max": "anticyclone", "min": "cyclone"}


# ------------------------------------------------------------------------------------------------- distances

def _augmented(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """Cost matrix of a diagram matching with the diagonal: L-infinity between points, persistence / 2 to it."""
    n, m = len(d1), len(d2)
    c = np.maximum(np.abs(d1[:, None, 0] - d2[None, :, 0]), np.abs(d1[:, None, 1] - d2[None, :, 1]))
    top, bot = np.full((n, n), np.inf), np.full((m, m), np.inf)
    np.fill_diagonal(top, (d1[:, 0] - d1[:, 1]) / 2)
    np.fill_diagonal(bot, (d2[:, 0] - d2[:, 1]) / 2)
    return np.block([[c, top], [bot, np.zeros((m, n))]])


def wasserstein(d1: np.ndarray, d2: np.ndarray, p: float = 1.0) -> float:
    if len(d1) + len(d2) == 0:
        return 0.0
    m = _augmented(d1, d2)
    m = np.where(np.isfinite(m), m, 1e3 * (m[np.isfinite(m)].max() + 1.0)) ** p
    r, c = linear_sum_assignment(m)
    return float(m[r, c].sum() ** (1 / p))


def bottleneck(d1: np.ndarray, d2: np.ndarray) -> float:
    if len(d1) + len(d2) == 0:
        return 0.0
    m = _augmented(d1, d2)
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
    """A regular (ny, nx) scoring grid: 1D latitude and longitude, and a map from flat emulator fields to images."""
    latitude: np.ndarray
    longitude: np.ndarray
    average: csr_matrix  # (ny * nx, n_fine): image cell mean of the fine cells it keeps
    mask: np.ndarray  # (ny, nx) cells with a value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """(..., n_fine) -> (..., ny, nx), NaN outside the mask."""
        flat = (self.average @ x.reshape(-1, x.shape[-1]).T).T.reshape(*x.shape[:-1], *self.mask.shape)
        return np.where(self.mask, flat, np.nan)

    @property
    def cell_km2(self) -> np.ndarray:
        dlat, dlon = np.diff(self.latitude).mean(), np.diff(self.longitude).mean()
        return EARTH_KM_PER_DEG**2 * dlat * dlon * np.cos(np.deg2rad(self.latitude))[:, None] * np.ones(self.mask.shape)


def quarter(lon: np.ndarray, lat: np.ndarray, region: np.ndarray, deg: float = QUARTER_DEG) -> Grid:
    """lon, lat, region: flat fine-grid arrays. Coarse cell centres at k * deg + deg / 2 over the fine grid's extent."""
    ix, iy = np.floor(lon / deg).astype(int), np.floor(lat / deg).astype(int)
    x0, y0 = ix.min(), iy.min()
    nx, ny = ix.max() - x0 + 1, iy.max() - y0 + 1
    cell = (iy - y0) * nx + (ix - x0)
    total = np.bincount(cell, minlength=nx * ny)
    inside = np.bincount(cell[region], minlength=nx * ny)
    keep = (inside > 0) & (2 * inside >= total)
    use = np.flatnonzero(region & keep[cell])
    count = np.bincount(cell[use], minlength=nx * ny)
    average = csr_matrix((1.0 / count[cell[use]], (cell[use], use)), shape=(nx * ny, len(lon)))
    return Grid((np.arange(ny) + y0 + 0.5) * deg, (np.arange(nx) + x0 + 0.5) * deg, average, keep.reshape(ny, nx))


def native(ds: xr.Dataset, region: np.ndarray) -> Grid:
    """The emulator grid itself: grid_index = ix * ny + iy -> image (iy, ix)."""
    nx, ny = int(ds.attrs["nx"]), int(ds.attrs["ny"])
    lon, lat = ds.lon.values.reshape(nx, ny), ds.lat.values.reshape(nx, ny)
    image = np.arange(nx * ny).reshape(nx, ny).T.ravel()  # image cell k holds grid_index image[k]
    average = csr_matrix((np.ones(nx * ny), (np.arange(nx * ny), image)), shape=(nx * ny, nx * ny))
    return Grid(lat[0], lon[:, 0], average, region.reshape(nx, ny).T)


def grids(ds: xr.Dataset, use_native: bool = False) -> tuple[Grid, Grid]:
    """(census grid, Loop Current grid) for a fields file."""
    region = ds.region.values.astype(bool)
    deep = region & (np.nan_to_num(ds.depth.values) > MIN_DEPTH_M)
    lc = quarter(ds.lon.values, ds.lat.values, region)
    return (native(ds, deep) if use_native else quarter(ds.lon.values, ds.lat.values, deep)), lc


# ----------------------------------------------------------------------------------------------- noise floor

def noise_floor(grid: Grid, n: int = 200, q: float = 0.95, sigma_m: float = NOISE_M, length_km: float = NOISE_KM,
                seed: int = 0) -> dict:
    """persistence-ocean's pure-noise null (experiment 02) on this grid: the q quantile of the persistence of every
    feature of correlated Gaussian noise, 4-connectivity, open boundary. A worst case: on a real sloped field noise
    rarely makes a new extremum, which is why the signal-plus-noise floor (EDDY_FLOOR_M) is lower."""
    noise = tp.spatial_surrogates(sigma_m, grid.latitude, grid.longitude, length_km, n, seed)
    p = np.concatenate([pairs.persistence for x in noise
                        for pairs in tp.compute_eddy_persistence(np.where(grid.mask, x, np.nan), open_boundary=True).values()])
    return {"floor_m": float(np.quantile(p, q)), "q": q, "n_fields": n, "features_per_field": len(p) / n,
            "cells": int(grid.mask.sum()), "sigma_m": sigma_m, "length_km": length_km}


# ---------------------------------------------------------------------------------------------- Loop Current

def lc_grid(grid: Grid) -> tp.GulfGrid:
    return tp.gulf_grid(grid.mask, grid.latitude, grid.longitude)


def lc_events(ssh: np.ndarray, dates, gg: tp.GulfGrid, km2: np.ndarray, floor: float = LC_FLOOR_M,
              first_back: pd.Timestamp | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Detachments and separation events of an SSH image series (T, ny, nx), each event with its neck track."""
    an = tp.gulf_anomaly(ssh, dates, gg)
    det = tp.lc17_detachments(an, gg, tp.TouchRule(floor, TOUCH_DAYS))
    ev = tp.separation_events(det)
    out = []
    if len(ev):
        maxima = tp.gulf_maxima(an, gg)
        previous = first_back if first_back is not None else an.dates[0]
        for _, e in ev.iterrows():
            earliest = max(previous, e["separation_date"] - pd.Timedelta(days=BACK_TRACK_MAX_DAYS))
            track = tp.follow_lce_back(maxima, e, earliest)
            row = det[det["date"] == e["separation_date"]]
            piece = row[(row.piece_max_lat == e["piece_max_lat"]) & (row.piece_max_lon == e["piece_max_lon"])].iloc[0]
            labels, _ = tp.lc17_labels(an.values[int(e["day"])], gg)
            out.append({"date": str(e["separation_date"].date()), "day": int(e["day"]), "censored": bool(e["censored"]),
                        "piece_lat": float(e["piece_max_lat"]), "piece_lon": float(e["piece_max_lon"]),
                        "piece_km2": float(km2[labels == labels[int(piece.piece_max_i), int(piece.piece_max_j)]].sum()),
                        "piece_h_max": float(piece.piece_h_max), "n_reattachments": int(e["n_reattachments"]),
                        "neck": {int(k): float(v) for k, v in zip(track["days_from_separation"], track["neck_depth"])}})
            if not e["censored"]:
                previous = e["separation_date"]
    return det, out


# --------------------------------------------------------------------------------------------------- scoring

def census(p: tp.Pairs, floor: float) -> dict:
    a = p.persistence[p.persistence >= floor]
    return {"count": len(a), "amp_median": float(np.median(a)) if len(a) else np.nan,
            "amp_p90": float(np.quantile(a, 0.9)) if len(a) else np.nan, "amp_sum": float(a.sum())}


def score_pair(model: np.ndarray, truth: np.ndarray, start: np.ndarray, floor: float) -> dict:
    """Census of model and truth images, and diagram distances model-truth and start-truth, per polarity."""
    pm, pt, ps = (tp.compute_eddy_persistence(x, **CENSUS) for x in (model, truth, start))
    out = {}
    for k, pol in POLARITY.items():
        for who, p in (("model", pm[pol]), ("truth", pt[pol])):
            out |= {f"{k}_{c}_{who}": v for c, v in census(p, floor).items()}
        dm, dt, ds = (tp.diagram(p[pol], pol, floor) for p in (pm, pt, ps))
        out[f"{k}_w1_model"], out[f"{k}_w1_pers"] = wasserstein(dm, dt), wasserstein(ds, dt)
        out[f"{k}_bottleneck_model"], out[f"{k}_bottleneck_pers"] = bottleneck(dm, dt), bottleneck(ds, dt)
    return out


def demean(img: np.ndarray) -> np.ndarray:
    """Minus the plain mean over the image's cells, per leading index (persistence does not see it; the LC does
    its own Gulf demeaning)."""
    return img - np.nanmean(img, axis=(-2, -1), keepdims=True)


def score(fields: Path, out: Path, use_native: bool = False, floor: float | None = None) -> xr.Dataset:
    ds = xr.open_dataset(fields)
    cg, lg = grids(ds, use_native)
    floor = (EDDY_FLOOR_M if not use_native else noise_floor(cg, 100)["floor_m"]) if floor is None else floor
    gg, km2 = lc_grid(lg), lg.cell_km2
    rows, events = [], []
    for i, s in enumerate(ds.start.values.astype("datetime64[D]")):
        h = int(ds.horizon.values[i])
        x0 = ds.ssh_start.values[i].astype(float)
        model, truth = (ds[f"ssh_{k}"].values[i, :h].astype(float) for k in ("model", "truth"))
        m, t, z = demean(cg(model)), demean(cg(truth)), demean(cg(x0))
        rows.append([score_pair(m[l], t[l], z, floor) for l in range(h)])
        dates = pd.date_range(str(s), periods=h + 1)
        for who, seq in (("model", model), ("truth", truth)):
            _, evs = lc_events(lg(np.concatenate([x0[None], seq])), dates, gg, km2)
            events += [{"start": str(s), "who": who, "lead": e["day"]} | e for e in evs]
    keys = list(rows[0][0])
    lead = np.arange(1, int(ds.horizon.max()) + 1)
    data = {k: (("start", "lead"), np.array([[r[l][k] if l < len(r) else np.nan for l in range(len(lead))] for r in rows]))
            for k in keys}
    res = xr.Dataset(data, {"start": ds.start.values, "lead": lead},
                     {"caveat": CAVEAT, "reference": REFERENCE, "grid": "native" if use_native else "0.25 deg DUACS cells",
                      "eddy_floor_m": floor, "lc_floor_m": LC_FLOOR_M, "cells": int(cg.mask.sum()), "fields": str(fields),
                      **{f"fields_{k}": str(v) for k, v in ds.attrs.items() if k in ("checkpoint", "checkpoint_md5", "code_sha")}})
    out.mkdir(parents=True, exist_ok=True)
    res.to_netcdf(out / "structure.nc")
    (out / "events.json").write_text(json.dumps({"caveat": CAVEAT, "reference": REFERENCE, "events": events}, indent=1))
    return res


def truth_series(pack: Path, out: Path, t_from: str, t_to: str) -> list[dict]:
    """Census counts and Loop Current events of the pack's own SSH (the reanalysis truth) over [t_from, t_to]."""
    meta = xr.open_zarr(pack / "meta.zarr", consolidated=True)
    names = [str(n) for n in meta.state_feature.values]
    st = lambda f: meta.static.sel(static_feature=f).values  # noqa: E731
    region = st("gulf").astype(bool) & st("ocean").astype(bool) & ~meta.boundary_mask.values.astype(bool)
    fake = xr.Dataset(coords={"lon": ("g", st("lon")), "lat": ("g", st("lat")), "region": ("g", region), "depth": ("g", st("depth"))})
    cg, lg = grids(fake)
    times = meta.time.values.astype("datetime64[D]")
    rows = np.flatnonzero((times >= np.datetime64(t_from)) & (times <= np.datetime64(t_to)))
    state = np.load(pack / "state.npy", mmap_mode="r")
    if state.dtype != np.float32:
        raise ValueError(f"{pack}/state.npy is {state.dtype}; truth reads a float32 pack")
    ssh = np.stack([np.asarray(state[t, :, names.index("ssh")], dtype=float) for t in rows])
    _, events = lc_events(lg(ssh), pd.DatetimeIndex(times[rows]), lc_grid(lg), lg.cell_km2)
    counts = [{k: census(p, EDDY_FLOOR_M)["count"] for k, p in tp.compute_eddy_persistence(x, **CENSUS).items()}
              for x in demean(cg(ssh))]
    out.mkdir(parents=True, exist_ok=True)
    (out / "truth_events.json").write_text(json.dumps({"caveat": "reanalysis truth, no model", "reference": REFERENCE,
                                                       "from": t_from, "to": t_to, "lc_floor_m": LC_FLOOR_M, "events": events}, indent=1))
    xr.Dataset({f"{k}_count": ("time", [c[k] for c in counts]) for k in ("anticyclone", "cyclone")},
               {"time": times[rows].astype("datetime64[ns]")}, {"eddy_floor_m": EDDY_FLOOR_M}).to_netcdf(out / "truth_series.nc")
    return events


MATCH_DAYS = 15  # a model separation within this many days of a truth one is the same event


def match_events(events: list[dict]) -> list[dict]:
    """Per start, each final truth separation and the nearest final model separation within MATCH_DAYS (or none),
    then the model's unmatched ones."""
    rows = []
    for start in sorted({e["start"] for e in events}):
        final = [e for e in events if e["start"] == start and not e["censored"]]
        truth, model = [e for e in final if e["who"] == "truth"], [e for e in final if e["who"] == "model"]
        used = set()
        for t in truth:
            near = sorted((abs(m["lead"] - t["lead"]), i) for i, m in enumerate(model) if i not in used)
            m = model[near[0][1]] if near and near[0][0] <= MATCH_DAYS else None
            if m is not None:
                used.add(near[0][1])
            rows.append({"start": start, "truth": t, "model": m})
        rows += [{"start": start, "truth": None, "model": m} for i, m in enumerate(model) if i not in used]
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
    at = {l: i for i, l in enumerate(lead) if l in (1, 7, 14, 21, 30, 45, 60)}
    tab = ["| model / truth, lead (d) | " + " | ".join(map(str, at)) + " |", "|---" * (len(at) + 1) + "|"]
    mean = lambda v: res[v].mean("start")  # noqa: E731
    for k in ("max", "min"):
        for name, v in ((f"{k} count", mean(f"{k}_count_model") / mean(f"{k}_count_truth")),
                        (f"{k} summed persistence", mean(f"{k}_amp_sum_model") / mean(f"{k}_amp_sum_truth")),
                        (f"{k} W1 model / W1 persistence", mean(f"{k}_w1_model") / mean(f"{k}_w1_pers"))):
            tab.append(f"| {name} | " + " | ".join(f"{float(v[i]):.2f}" for i in at.values()) + " |")
    rows = match_events(events)
    f = lambda x, s=1: "-" if x is None else f"{x * s:.0f}"  # noqa: E731
    get = lambda e, k: None if e is None else e[k]  # noqa: E731
    neck0 = lambda e: None if e is None else e["neck"].get("0", e["neck"].get(0))  # noqa: E731
    ev = ["| start | truth lead | model lead | error (d) | neck at separation, truth / model (cm) | piece area, truth / model (1000 km2) |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        t, m = r["truth"], r["model"]
        err = m["lead"] - t["lead"] if t and m else None
        ev.append(f"| {r['start']} | {f(get(t, 'lead'))} | {f(get(m, 'lead'))} | {f(err)} | {f(neck0(t), 100)} / {f(neck0(m), 100)} "
                  f"| {f(get(t, 'piece_km2'), 1e-3)} / {f(get(m, 'piece_km2'), 1e-3)} |")
    ok = np.array([r["model"]["lead"] - r["truth"]["lead"] for r in rows if r["truth"] and r["model"]], float)
    n_t, n_m = sum(r["truth"] is not None for r in rows), sum(r["model"] is not None for r in rows)
    cens = {w: sum(e["censored"] and e["who"] == w for e in events) for w in ("truth", "model")}
    md = "\n".join([f"# {title}", "", f"**Caveat.** {CAVEAT}.", "", f"Engine: {REFERENCE}.", "",
                    f"Census grid: {res.grid} ({res.cells} cells, water deeper than {MIN_DEPTH_M:.0f} m), floor "
                    f"{res.eddy_floor_m * 100:.1f} cm; LC touch floor {res.lc_floor_m * 100:.2f} cm; {res.sizes['start']} starts, "
                    f"leads 1-{lead[-1]}. Ratios are of means over starts.", "", *tab, "",
                    f"Final LC separations inside the runs (off LC17 more than 30 days, no counted rejoin): truth {n_t}, model {n_m}, "
                    f"matched within {MATCH_DAYS} days {len(ok)}"
                    + (f", median |error| {np.median(np.abs(ok)):.0f} d, mean {ok.mean():+.1f} d" if len(ok) else "")
                    + f". Censored by the run's end (undecided): truth {cens['truth']}, model {cens['model']}.", "", *ev, "",
                    "![structure](structure.png)", ""])
    (score_dir / "report.md").write_text(md)
    return md


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("score")
    q.add_argument("fields", type=Path)
    q.add_argument("out", type=Path)
    q.add_argument("--native", action="store_true", help="census on the emulator grid with a recalibrated pure-noise floor")
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
    elif a.cmd == "truth":
        print(json.dumps(truth_series(a.pack, a.out, a.t_from, a.t_to), indent=1))
    elif a.cmd == "report":
        print(report(a.score_dir, a.title))
    else:
        ds = xr.open_dataset(a.fields)
        res = {name: noise_floor(grids(ds, nat)[0], a.n) for name, nat in (("quarter", False), ("native", True))}
        a.out.write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
