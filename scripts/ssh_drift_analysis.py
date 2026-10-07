"""Evidence tables for the SSH drift of the reanalysis emulator (docs/conservation.md), from conservation outputs.

    python scripts/ssh_drift_analysis.py <probe_dir> <nlam.yaml>

<probe_dir> holds `conservation forecast` outputs (<ckpt>_<split>.npz/.json) and `conservation series` output
(truth_series.npz). Prints: per split and checkpoint, the mean error and one-step change of each quantity; the
truth's Gulf-mean SSH change per calendar month, train years against the test year; the lead-1 SSH error against
the initial state's Gulf-mean SSH anomaly; and the mean lead-1 SSH error by distance from the boundary band.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage

from hycom_emulator.evaluate_rea import cell_area, pack_path, regions

# Source experiment of the 3z (T, S, u, v) rows by start date, docs/rea-pipeline.md; 2d (SSH) differs in Jan 2024.
EXPERIMENTS = (("020", "2001-01-01"), ("031", "2017-06-02"), ("035", "2021-01-01"), ("037", "2024-01-02"), ("038", "2024-04-02"))
QUANTITIES = ("ssh_mean", "ubaro_mean", "vbaro_mean", "temp_mean", "salin_mean", "u_mean", "v_mean", "heat_content", "salt_content")


def forecasts_table(d: Path) -> None:
    print("== mean error per lead (pred - truth) and mean one-step change (truth | model), lead 1")
    for path in sorted(d.glob("s[0-9]_*.json")):
        s = json.loads(path.read_text())
        for region in ("gulf", "interior"):
            q = s["regions"][region]
            for name in QUANTITIES:
                leads = q[name]["leads"]
                err = " ".join(f"{v['error_mean']:+.4g}" for v in leads.values())
                one = leads["1"]
                print(f"{path.stem:9s} n={s['samples']:3d} {region:8s} {name:13s} err {err}  sd1 {one['error_std']:.3g}  "
                      f"pos1 {one['positive_fraction']:.2f}  step1 truth {one['truth_step']:+.4g} model {one['model_step']:+.4g}")


def truth_months(d: Path) -> None:
    z = np.load(d / "truth_series.npz")
    t, v = z["time"], z["values"]
    r, q = list(z["regions"]).index("gulf"), list(z["quantities"]).index("ssh_mean")
    ssh = v[:, r, q]
    dt = np.diff(ssh)
    ok = ~(z["time_filled"][1:] | z["time_filled"][:-1])
    month = (t[1:].astype("datetime64[M]").astype(int) % 12) + 1
    year = t[1:].astype("datetime64[Y]").astype(int) + 1970
    print("== truth Gulf-mean SSH change per day (cm/day), by month: train 2001-2021 | val 2022-2023 | test 2024")
    for m in range(1, 13):
        cols = []
        for lo, hi in ((2001, 2021), (2022, 2023), (2024, 2024)):
            sel = ok & (month == m) & (year >= lo) & (year <= hi)
            cols.append(f"{100 * dt[sel].mean():+.3f} (sd {100 * dt[sel].std():.2f}, n {sel.sum()})" if sel.any() else "-")
        print(f"month {m:2d}: " + " | ".join(cols))
    for lo, hi in ((2001, 2021), (2022, 2023), (2024, 2024)):
        sel = ok & (year >= lo) & (year <= hi)
        print(f"{lo}-{hi}: mean change {100 * dt[sel].mean():+.4f} cm/day, sd {100 * dt[sel].std():.3f}, "
              f"mean level {100 * ssh[1:][sel].mean():+.2f} cm")
    print("== Gulf-mean SSH level by year (cm):", " ".join(f"{y}:{100 * ssh[1:][year == y].mean():+.1f}" for y in np.unique(year)))


def scales(d: Path) -> None:
    """Natural size of each quantity's one-day change on train days: the scale a domain-mean loss term divides by."""
    z = np.load(d / "truth_series.npz")
    ok = ~(z["time_filled"][1:] | z["time_filled"][:-1]) & (z["time"][1:] < np.datetime64("2022-01-01"))
    step = np.diff(z["values"], axis=0)[ok]
    print("== train one-day change of each quantity: mean | sd (units of the quantity per day)")
    for r, region in enumerate(z["regions"]):
        print(f"{region:8s} " + "; ".join(f"{q} {step[:, r, k].mean():+.3g}|{step[:, r, k].std():.3g}" for k, q in enumerate(z["quantities"])))


def ts_residual(z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gulf-mean SSH minus its linear fit on Gulf-mean T and S over train days: (coef, residual per row, train rows)."""
    q, r = list(z["quantities"]), list(z["regions"]).index("gulf")
    v, t = z["values"][:, r], z["time"]
    x = np.column_stack([np.ones(len(t)), v[:, q.index("temp_mean")], v[:, q.index("salin_mean")]])
    train = (t < np.datetime64("2022-01-01")) & ~z["time_filled"]
    coef, *_ = np.linalg.lstsq(x[train], v[train, q.index("ssh_mean")], rcond=None)
    return coef, v[:, q.index("ssh_mean")] - x @ coef, train


def level_vs_density(d: Path) -> None:
    """Gulf-mean SSH against Gulf-mean T and S content: a linear fit on train days, then its residual per year and,
    in 2023-2024, per month; plus the one-day SSH jumps on the days the source experiment changes."""
    z = np.load(d / "truth_series.npz")
    t, ssh = z["time"], z["values"][:, list(z["regions"]).index("gulf"), list(z["quantities"]).index("ssh_mean")]
    coef, res, train = ts_residual(z)
    year = t.astype("datetime64[Y]").astype(int) + 1970
    print(f"== Gulf SSH = {coef[0]:+.3f} {coef[1]:+.4f} T {coef[2]:+.4f} S (train fit, r2 {1 - res[train].var() / ssh[train].var():.2f}); residual (cm) per year")
    print(" ".join(f"{y}:{100 * res[year == y].mean():+.1f}" for y in np.unique(year)))
    ym = t.astype("datetime64[M]")
    print("residual per month 2023-2024:", " ".join(f"{str(m)[2:]}:{100 * res[ym == m].mean():+.1f}" for m in np.unique(ym[year >= 2023])))
    for day in ("2017-06-01", "2017-06-02", "2021-01-01", "2024-01-01", "2024-01-02", "2024-01-06", "2024-02-02", "2024-04-02"):
        i = int(np.flatnonzero(t.astype("datetime64[D]") == np.datetime64(day))[0])
        print(f"{day}: Gulf SSH change {100 * (ssh[i] - ssh[i - 1]):+.2f} cm, T-S residual change {100 * (res[i] - res[i - 1]):+.2f} cm")


def against_anomaly(d: Path) -> None:
    print("== lead-1 Gulf SSH-mean error (cm) against the t0 Gulf-mean SSH relative to the train mean of that calendar month")
    z = np.load(d / "truth_series.npz")
    r, q = list(z["regions"]).index("gulf"), list(z["quantities"]).index("ssh_mean")
    month = z["time"].astype("datetime64[M]").astype(int) % 12
    train = z["time"] < np.datetime64("2022-01-01")
    clim = np.array([z["values"][train & (month == m), r, q].mean() for m in range(12)])
    _, res, _ = ts_residual(z)
    pooled = []
    for path in sorted(d.glob("s[0-9]_*.npz")):
        f = np.load(path)
        rr, qq = list(f["regions"]).index("gulf"), list(f["quantities"]).index("ssh_mean")
        anom = f["true"][:, rr, 0, qq] - clim[f["t0"].astype("datetime64[M]").astype(int) % 12]
        err = f["pred"][:, rr, 1, qq] - f["true"][:, rr, 1, qq]
        step = f["true"][:, rr, 1, qq] - f["true"][:, rr, 0, qq]
        slope, icpt = np.polyfit(anom, err, 1)
        month = f["t0"].astype("datetime64[M]").astype(int) % 12 + 1
        for name, scale, unit in (("ssh_mean", 100, "cm"), ("temp_mean", 1000, "mK"), ("v_mean", 100, "cm/s")):
            j = list(f["quantities"]).index(name)
            e1 = f["pred"][:, rr, 1, j] - f["true"][:, rr, 1, j]
            by = " ".join(f"{m}:{scale * e1[month == m].mean():+.2f}" for m in np.unique(month))
            print(f"{path.stem:9s} lead-1 {name} error by start month ({unit}): {by}")
            exp = np.searchsorted(np.array([np.datetime64(e[1]) for e in EXPERIMENTS]), f["t0"].astype("datetime64[D]"), side="right") - 1
            by = " ".join(f"{EXPERIMENTS[x][0]}:{scale * e1[exp == x].mean():+.2f} (n {np.sum(exp == x)})" for x in np.unique(exp))
            print(f"{path.stem:9s} lead-1 {name} error by experiment ({unit}): {by}")
        days = z["time"].astype("datetime64[D]")
        res0 = res[np.searchsorted(days, f["t0"].astype("datetime64[D]"))]
        model_step = f["pred"][:, rr, 1, qq] - f["true"][:, rr, 0, qq]
        pooled.append((res0, err, model_step, step))
        k, c = np.polyfit(res0, err, 1)
        print(f"{path.stem:9s} t0 T-S residual mean {100 * res0.mean():+.2f} cm sd {100 * res0.std():.2f}; lead-1 error = {100 * c:+.3f} cm "
              f"{k:+.4f} x residual, corr {np.corrcoef(res0, err)[0, 1]:+.2f}; corr(model step, residual) {np.corrcoef(res0, model_step)[0, 1]:+.2f}, "
              f"corr(truth step, residual) {np.corrcoef(res0, step)[0, 1]:+.2f}")
        print(f"{path.stem:9s} anomaly mean {100 * anom.mean():+.2f} cm sd {100 * anom.std():.2f}; error = {icpt * 100:+.3f} cm "
              f"{slope:+.4f} x anomaly; corr(err, anomaly) {np.corrcoef(anom, err)[0, 1]:+.2f}, corr(err, truth step) {np.corrcoef(step, err)[0, 1]:+.2f}")
    print("== s2 forecasts of all splits pooled, against the t0 T-S residual")
    _pooled_fit([p for p, path in zip(pooled, sorted(d.glob("s[0-9]_*.npz"))) if path.stem.startswith("s2_")])


def _pooled_fit(pooled: list) -> None:
    res0, err, model_step, step = (np.concatenate(a) for a in zip(*pooled))
    for name, y in (("lead-1 error", err), ("model step", model_step), ("truth step", step)):
        k, c = np.polyfit(res0, y, 1)
        print(f"pooled n={res0.size}: {name} = {100 * c:+.3f} cm {k:+.4f} x t0 residual, corr {np.corrcoef(res0, y)[0, 1]:+.2f}")


def projected(d: Path) -> None:
    """A uniform SSH shift changes only the offset, so the Gulf SSH RMSE of a forecast whose Gulf-mean SSH follows
    x0 plus a prescribed mean change per day is sqrt(mean(new_offset^2 + pattern^2)) over forecasts."""
    print("== Gulf SSH RMSE (cm) per lead: model | Gulf mean held at x0 | Gulf mean + train climatological change of the month")
    z = np.load(d / "truth_series.npz")
    r, q = list(z["regions"]).index("gulf"), list(z["quantities"]).index("ssh_mean")
    ssh, t = z["values"][:, r, q], z["time"]
    ok = ~(z["time_filled"][1:] | z["time_filled"][:-1]) & (t[1:] < np.datetime64("2022-01-01"))
    month = t[1:].astype("datetime64[M]").astype(int) % 12
    clim = np.array([np.diff(ssh)[ok & (month == m)].mean() for m in range(12)])
    for path in sorted(d.glob("s[0-9]_*.npz")):
        f = np.load(path)
        rr, qq = list(f["regions"]).index("gulf"), list(f["quantities"]).index("ssh_mean")
        true, x0 = f["true"][:, rr, :, qq], f["true"][:, rr, 0, qq]
        pattern, offset = f["ssh_pattern"][:, rr], f["ssh_offset"][:, rr]
        lead = np.arange(1, true.shape[1])
        step = clim[f["t0"].astype("datetime64[M]").astype(int) % 12][:, None]
        rows = {"perfect mean": np.zeros_like(offset), "model": offset, "held": x0[:, None] - true[:, 1:], "clim": x0[:, None] + lead * step - true[:, 1:]}
        val = d / f"{path.stem.split('_')[0]}_val.json"
        if path.stem.endswith("_test") and val.is_file():
            bias = [v["offset_mean"] for v in json.loads(val.read_text())["regions"]["gulf"]["ssh_split"].values()]
            rows["model - val offset"] = offset - np.array(bias)
        out = " | ".join(f"{k} " + " ".join(f"{100 * np.sqrt(np.mean(o**2 + pattern**2, 0))[l]:.3f}" for l in range(len(lead))) for k, o in rows.items())
        print(f"{path.stem:9s} {out}")


def loss_domain(d: Path, config: Path) -> None:
    """Per channel, lead-1 mean error in change stds: over the points the loss counts (fill included) and over real ones.
    With a free output bias per channel, a converged MSE fit drives the first to zero on train data, not the second."""
    meta = xr.open_zarr(pack_path(config) / "meta.zarr", consolidated=True).load()
    names = [str(n) for n in meta.state_feature.values]
    pick = ["ssh", "ubaro", "vbaro"] + [f"{v}_{z}m" for v in ("v", "u", "temp") for z in (0, 100, 500, 1000, 2000)]
    print("== lead-1 mean error / change std: loss domain | real points (fill share of the loss domain)")
    inside = ~meta.boundary_mask.values.astype(bool)
    lev = list(meta.level.values.astype(float))
    for path in sorted(d.glob("s[0-9]_*.npz")):
        f = np.load(path)
        if "loss_domain_bias" not in f:
            continue
        row = []
        for n in pick:
            j = names.index(n)
            var, _, z = n.rpartition("_")
            fill = 1 - meta.level_ocean.values[inside, lev.index(float(z[:-1]))].mean() if var else 0.0
            row.append(f"{n} {f['loss_domain_bias'][0, j]:+.4f}|{f['real_point_bias'][0, j]:+.4f} ({fill:.2f})")
        print(f"{path.stem:9s} " + "; ".join(row))


def by_band_distance(d: Path, config: Path) -> None:
    meta = xr.open_zarr(pack_path(config) / "meta.zarr", consolidated=True).load()
    nx, ny = np.unique(meta.x.values).size, np.unique(meta.y.values).size
    band = meta.boundary_mask.values.astype(bool) & meta.static.sel(static_feature="ocean").values.astype(bool)
    dist = ndimage.distance_transform_edt(~band.reshape(nx, ny)).reshape(-1)
    area, reg = cell_area(meta), regions(meta)
    print("== mean lead-1 and lead-4 SSH error (cm) by distance from the boundary band (grid cells); share of positive points")
    for path in sorted(d.glob("s[0-9]_*.npz")):
        e = np.load(path)["ssh_error_map"]
        for lo, hi in ((0, 5), (5, 15), (15, 40), (40, 1e9)):
            m = reg["interior"] & (dist > lo) & (dist <= hi)
            w = area * m
            print(f"{path.stem:9s} dist ({lo:g},{hi:g}] n={m.sum():5d} lead1 {100 * (w * e[0]).sum() / w.sum():+.3f} lead4 "
                  f"{100 * (w * e[-1]).sum() / w.sum():+.3f} pos1 {(e[0][m] > 0).mean():.2f}")
        g = reg["gulf"]
        print(f"{path.stem:9s} gulf: share of points with positive mean error, lead 1 {(e[0][g] > 0).mean():.2f}, lead 4 {(e[-1][g] > 0).mean():.2f}")


if __name__ == "__main__":
    d, cfg = Path(sys.argv[1]), Path(sys.argv[2])
    forecasts_table(d)
    if (d / "truth_series.npz").is_file():
        truth_months(d)
        scales(d)
        level_vs_density(d)
        against_anomaly(d)
        projected(d)
    loss_domain(d, cfg)
    by_band_distance(d, cfg)
