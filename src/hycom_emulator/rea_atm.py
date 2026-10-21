"""Daily atmospheric forcing channels for the reanalysis pack, from the CFSR and CFSv2 Gulf subsets (card emu-rea-atm-prep).

    python -m hycom_emulator.rea_atm extract <tmp_dir> --years 2000-2024
    python -m hycom_emulator.rea_atm wind <pack> <tmp_dir> <out_dir> [--years 2001-2024]
    python -m hycom_emulator.rea_atm build <pack> <tmp_dir> <out_dir> [--years 2001-2024]
    python -m hycom_emulator.rea_atm check <pack> <out_dir>

`extract` unpacks the six hourly variables of the asked years from the two tarballs into tmp_dir. `wind` compares the
CFS daily 10 m wind with the pack's own wnd_ewd and wnd_nwd for both averaging windows, per product era, writes
checks/wind_check.json and picks the window. `build` writes atm.npy (pack rows x pack points x CHANNELS, float32, NaN
where not built) on that window and meta.json. `check` writes the 2010-2011 seam test and the monthly climatology.

Each hourly field is an instantaneous value (time_average_bounds 0, 0) on the product's regular Gulf grid. A day's mean
uses the hours of that day only:
    a  00Z..23Z, equal weights;
    b  as the reanalysis daily files (ncra --wgt 0.92,1,...,1,0.92 over hours 12..23, 00..11, 12 of the same day):
       00Z..23Z with 12Z weighted 1.84.
Values are regridded bilinearly to the pack's stride-2 points; a point outside a product's grid is an error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from hycom_emulator.evaluate_rea import regions

TARBALLS = {"cfsr": Path("/hycom/ftp/pub/GitHub/HYCOM-examples/datasets/CFSR_GOM.tar.gz"),
            "cfsv2": Path("/hycom/ftp/pub/GitHub/HYCOM-examples/datasets/CFSv2_GOM.tar.gz")}
FOLDERS = {"cfsr": "CFSR_GOM", "cfsv2": "CFSv2_GOM"}
FILES = {"Taqa": "sea", "dswsfc": "sec", "dlwsfc": "sec", "precip": "sea", "sfcprs": "sec", "uv-10m": "sec2"}
CHANNELS = ("airtmp", "vapmix", "dswflx", "dlwflx", "precip", "sfcprs")
WIND = ("wndewd", "wndnwd")
PACK_WIND = ("wnd_ewd", "wnd_nwd")
WINDOWS = {"a": np.ones(24), "b": np.where(np.arange(24) == 12, 1.84, 1.0)}
TRAIN_END = np.datetime64("2021-12-31")
SEAM = 2011
CORR_MIN, RMS_MAX = 0.97, 0.15


def product(year: int) -> str:
    return "cfsr" if year <= 2010 else "cfsv2"


def hourly_path(tmp: Path, year: int, tag: str) -> Path:
    p = product(year)
    return tmp / FOLDERS[p] / f"{p}_gom-{FILES[tag]}_{year}_01hr_{tag}.nc"


def years_arg(s: str) -> list[int]:
    """'2001,2010-2012' -> [2001, 2010, 2011, 2012]."""
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def extract(tmp: Path, years: list[int]) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    for p, tar in TARBALLS.items():
        want = [str(hourly_path(tmp, y, t).relative_to(tmp)) for y in years if product(y) == p for t in FILES]
        if want:
            subprocess.run(["tar", "xzf", str(tar), "-C", str(tmp), *want], check=True)
    missing = [str(hourly_path(tmp, y, t)) for y in years for t in FILES if not hourly_path(tmp, y, t).exists()]
    if missing:
        raise FileNotFoundError(f"not extracted: {missing}")


@dataclass(frozen=True)
class Bilinear:
    """Weights from a regular (lat, lon) grid to points: values(..., ny, nx) -> (..., npoints)."""
    i0: np.ndarray
    j0: np.ndarray
    wy: np.ndarray
    wx: np.ndarray

    @classmethod
    def make(cls, lat: np.ndarray, lon: np.ndarray, plat: np.ndarray, plon: np.ndarray) -> Bilinear:
        plon = np.where(plon < lon.min(), plon + 360, plon)
        if not (lat[0] <= plat.min() and plat.max() <= lat[-1] and lon[0] <= plon.min() and plon.max() <= lon[-1]):
            raise ValueError(f"pack points fall outside the product grid: lat {lat[0]}..{lat[-1]}, lon {lon[0]}..{lon[-1]}")
        i0 = np.clip(np.searchsorted(lat, plat) - 1, 0, len(lat) - 2)
        j0 = np.clip(np.searchsorted(lon, plon) - 1, 0, len(lon) - 2)
        return cls(i0, j0, (plat - lat[i0]) / (lat[i0 + 1] - lat[i0]), (plon - lon[j0]) / (lon[j0 + 1] - lon[j0]))

    def __call__(self, f: np.ndarray) -> np.ndarray:
        i, j, wy, wx = self.i0, self.j0, self.wy, self.wx
        return ((1 - wy) * (1 - wx) * f[..., i, j] + (1 - wy) * wx * f[..., i, j + 1]
                + wy * (1 - wx) * f[..., i + 1, j] + wy * wx * f[..., i + 1, j + 1])


def daily(path: Path, var: str, days: np.ndarray, window: str) -> tuple[np.ndarray, xr.Dataset]:
    """(len(days), ny, nx) float32 weighted daily means of var for days (datetime64[D]) and the opened dataset."""
    ds = xr.open_dataset(path)
    t = ds.MT.values.astype("datetime64[m]")
    day, hour = t.astype("datetime64[D]"), ((t - t.astype("datetime64[D]")) / np.timedelta64(1, "h")).round().astype(int)
    x = ds[var].values.astype(np.float32)
    w = WINDOWS[window]
    out = np.full((len(days), *x.shape[1:]), np.nan, np.float32)
    for k, d in enumerate(days):
        idx = np.flatnonzero(day == d)
        if len(idx) == 24 and (hour[idx] == np.arange(24)).all():
            out[k] = np.tensordot(w, x[idx], 1) / w.sum()
    return out, ds


def data_vars(ds: xr.Dataset) -> list[str]:
    return [v for v in ds.data_vars if ds[v].dims == ("MT", "Latitude", "Longitude")]


@dataclass
class Pack:
    dates: np.ndarray  # datetime64[D] per row
    lat: np.ndarray
    lon: np.ndarray
    gulf: np.ndarray
    wind: np.ndarray  # mmap (rows, grid, forcing_feature)
    wind_cols: list[int]

    @classmethod
    def open(cls, pack: Path) -> Pack:
        meta = xr.open_zarr(pack / "meta.zarr", consolidated=True).load()
        st = lambda k: meta.static.sel(static_feature=k).values  # noqa: E731
        names = [str(n) for n in meta.forcing_feature.values]
        return cls(meta.time.values.astype("datetime64[D]"), st("lat"), st("lon"), regions(meta)["gulf"],
                   np.load(pack / "forcing.npy", mmap_mode="r"), [names.index(n) for n in PACK_WIND])

    def rows(self, year: int) -> np.ndarray:
        return np.flatnonzero(self.dates.astype("datetime64[Y]").astype(int) + 1970 == year)


def wind(pack: Path, tmp: Path, out: Path, years: list[int]) -> dict:
    pk = Pack.open(pack)
    sums = {}  # (window, era, comp) -> [n, sx, sy, sxx, syy, sxy, sdd]
    for y in years:
        r = pk.rows(y)
        if not len(r):
            continue
        truth = np.asarray(pk.wind[r][:, pk.gulf][..., pk.wind_cols], np.float64)  # (days, gulf points, 2)
        for wname in WINDOWS:
            for c, v in enumerate(WIND):
                f, ds = daily(hourly_path(tmp, y, "uv-10m"), v, pk.dates[r], wname)
                interp = Bilinear.make(ds.Latitude.values, ds.Longitude.values, pk.lat[pk.gulf], pk.lon[pk.gulf])
                x, t = interp(f).astype(np.float64), truth[..., c]
                ok = np.isfinite(x) & np.isfinite(t)
                x, t = x[ok], t[ok]
                s = sums.setdefault((wname, product(y), PACK_WIND[c]), np.zeros(7))
                s += [x.size, x.sum(), t.sum(), (x * x).sum(), (t * t).sum(), (x * t).sum(), ((x - t) ** 2).sum()]
    res = {}
    for (wname, era, comp), (n, sx, st, sxx, stt, sxt, sdd) in sums.items():
        cov, vx, vt = sxt / n - sx * st / n**2, sxx / n - (sx / n) ** 2, stt / n - (st / n) ** 2
        res.setdefault(wname, {}).setdefault(era, {})[comp] = {
            "n": int(n), "corr": float(cov / np.sqrt(vx * vt)), "rms_over_std": float(np.sqrt(sdd / n) / np.sqrt(vt)),
            "mean_diff": float((sx - st) / n)}
    score = {w: np.mean([c["corr"] for e in v.values() for c in e.values()]) for w, v in res.items()}
    chosen = max(score, key=score.get)
    passed = {era: all(c["corr"] >= CORR_MIN and c["rms_over_std"] <= RMS_MAX for c in comps.values())
              for era, comps in res[chosen].items()}
    report = {"windows": res, "mean_corr": score, "chosen": chosen, "pass_by_era": passed,
              "thresholds": {"corr_min": CORR_MIN, "rms_over_std_max": RMS_MAX}, "years": years}
    (out / "checks").mkdir(parents=True, exist_ok=True)
    (out / "checks/wind_check.json").write_text(json.dumps(report, indent=1))
    return report


def build(pack: Path, tmp: Path, out: Path, years: list[int]) -> None:
    pk = Pack.open(pack)
    window = json.loads((out / "checks/wind_check.json").read_text())["chosen"]
    atm = np.lib.format.open_memmap(out / "atm.npy", mode="r+" if (out / "atm.npy").exists() else "w+", dtype=np.float32,
                                    shape=(len(pk.dates), len(pk.lat), len(CHANNELS)))
    if not (out / "atm.npy.init").exists():
        atm[:] = np.nan
        (out / "atm.npy.init").touch()
    meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {"years": {}}
    for y in years:
        r = pk.rows(y)
        if not len(r):
            continue
        files, units = {}, {}
        for tag in ("Taqa", "dswsfc", "dlwsfc", "precip", "sfcprs"):
            path = hourly_path(tmp, y, tag)
            names = data_vars(xr.open_dataset(path))
            for v in names:
                if v not in CHANNELS:
                    raise KeyError(f"{path.name} holds {v}, not one of {CHANNELS}")
                f, ds = daily(path, v, pk.dates[r], window)
                interp = Bilinear.make(ds.Latitude.values, ds.Longitude.values, pk.lat, pk.lon)
                vals, u = interp(f), ds[v].attrs.get("units", "")
                if v == "airtmp" and u.strip().lower() in ("k", "kelvin", "degk"):
                    vals, u = vals - 273.15, "degC"
                atm[r, :, CHANNELS.index(v)] = vals
                files[v], units[v] = path.name, u
        if set(files) != set(CHANNELS):
            raise KeyError(f"{y}: built {sorted(files)}, expected {CHANNELS}")
        meta["years"][str(y)] = {"product": product(y), "files": files}
        meta["units"] = units
        atm.flush()
    train = pk.dates <= TRAIN_END
    built = np.isfinite(atm[:, 0, 0])
    sel = np.flatnonzero(train & built)
    meta.update({"channels": list(CHANNELS), "window": window, "rows": int(len(pk.dates)), "rows_built": int(built.sum()),
                 "product_per_row": [product(int(str(d)[:4])) for d in pk.dates], "dates": [str(d) for d in pk.dates],
                 "train_end": str(TRAIN_END)})
    if len(sel):
        n, s1, s2 = np.zeros(len(CHANNELS)), np.zeros(len(CHANNELS)), np.zeros(len(CHANNELS))
        for i in range(0, len(sel), 64):
            x = np.asarray(atm[sel[i:i + 64]], np.float64).reshape(-1, len(CHANNELS))
            ok = np.isfinite(x)
            n, s1, s2 = n + ok.sum(0), s1 + np.where(ok, x, 0).sum(0), s2 + np.where(ok, x * x, 0).sum(0)
        m = s1 / n
        meta["train_mean"] = dict(zip(CHANNELS, m.tolist()))
        meta["train_std"] = dict(zip(CHANNELS, np.sqrt(s2 / n - m * m).tolist()))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def check(pack: Path, out: Path) -> dict:
    import pandas as pd
    pk = Pack.open(pack)
    atm = np.load(out / "atm.npy", mmap_mode="r")
    g = np.flatnonzero(pk.gulf)
    gm = np.concatenate([np.nanmean(np.asarray(atm[i:i + 64][:, g]), axis=1) for i in range(0, len(atm), 64)])  # (rows, channel)
    df = pd.DataFrame(gm, index=pd.DatetimeIndex(pk.dates), columns=CHANNELS)
    annual = df.groupby(df.index.year).mean()
    seam = {}
    for c in CHANNELS:
        a = annual[c].dropna()
        d = a.diff().dropna()
        other = d.drop(SEAM, errors="ignore")
        step = float(d.get(SEAM, np.nan))
        seam[c] = {"step_2010_2011": step, "std_other_steps": float(other.std()), "ratio": float(abs(step) / other.std()),
                   "pass": bool(abs(step) <= 2 * other.std())}
    clim = df[df.index <= pd.Timestamp(str(TRAIN_END))].groupby(df.index[df.index <= pd.Timestamp(str(TRAIN_END))].month).mean()
    y2024 = df[df.index.year == 2024].groupby(df[df.index.year == 2024].index.month).mean()
    res = {"seam": seam, "rows_finite": int(np.isfinite(gm).all(1).sum()), "rows": int(len(gm)),
           "climatology_2001_2021": clim.round(6).to_dict(), "monthly_2024": y2024.round(6).to_dict()}
    (out / "checks/seam.json").write_text(json.dumps(res, indent=1, default=float))
    lines = ["| month | " + " | ".join(f"{c} clim | {c} 2024" for c in CHANNELS) + " |", "|---|" + "---|" * (2 * len(CHANNELS))]
    for m in range(1, 13):
        cells = [f"{clim[c].get(m, np.nan):.4g} | {y2024[c].get(m, np.nan):.4g}" for c in CHANNELS]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    (out / "checks/climatology.md").write_text("\n".join(lines) + "\n")
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("tmp", type=Path)
    e.add_argument("--years", type=years_arg, default=years_arg("2000-2024"))
    for name in ("wind", "build"):
        s = sub.add_parser(name)
        s.add_argument("pack", type=Path)
        s.add_argument("tmp", type=Path)
        s.add_argument("out", type=Path)
        s.add_argument("--years", type=years_arg, default=years_arg("2001-2024"))
    c = sub.add_parser("check")
    c.add_argument("pack", type=Path)
    c.add_argument("out", type=Path)
    a = p.parse_args()
    if a.cmd == "extract":
        extract(a.tmp, a.years)
    elif a.cmd == "wind":
        a.out.mkdir(parents=True, exist_ok=True)
        print(json.dumps(wind(a.pack, a.tmp, a.out, a.years), indent=1))
    elif a.cmd == "build":
        build(a.pack, a.tmp, a.out, a.years)
    else:
        print(json.dumps(check(a.pack, a.out)["seam"], indent=1))


if __name__ == "__main__":
    main()
