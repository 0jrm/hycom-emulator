"""Conservation diagnostics of the reanalysis emulator: errors of domain means and column contents, per forecast and lead.

    python -m hycom_emulator.conservation forecast <nlam.yaml> <ckpt> <out_prefix> [--split test] [--ar-steps 4] [--limit n]
    python -m hycom_emulator.conservation series <nlam.yaml> <out_prefix>

Every quantity is a linear functional of the state, sum(w * x) over (grid point, channel):

- ssh_mean, ubaro_mean, vbaro_mean: area means (m, m/s).
- temp_mean, salin_mean, u_mean, v_mean: volume means to 2000 m (degC, psu, m/s).
- heat_content: rho0 cp times the integral of T over depth, as an area mean (J/m2).
- salt_content: rho0 times the integral of S/1000 over depth, as an area mean (kg/m2).

Weights are those of hycom_emulator.evaluate_rea: area cos^2(lat) on the region's points, times the depth interval
of each level on points real at that level (`level_ocean`). Regions: gulf and interior (all ocean outside the
boundary band). Since each quantity is linear, the error of the quantity is the quantity of the error.

`forecast` runs the model on a split. <out_prefix>.json holds, per region, quantity and lead, the mean and std of
the error over forecasts, the share of positive errors and the mean one-day change of truth and model.
<out_prefix>.npz holds every forecast's values, the SSH error split into area-mean offset and pattern RMSE, the
mean SSH error map, the area-mean squared error of pointwise column heat and salt content, and each channel's mean
error in change stds over the points neural-lam's loss counts (fill included) and over real points only.

`series` computes every quantity on every truth row of the pack, plus the area-mean square of the one-day change of
column heat and salt content over train rows: the scales of the loss terms in docs/conservation.md. It reads rows
with pread and drops them from the page cache, so memory stays flat.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hycom_emulator.evaluate_rea import cell_area, level_weights, pack_path, point_weights, regions

RHO0 = 1025.0  # kg/m3
CP = 3990.0  # J/(kg K), HYCOM's spcifh
SURFACE = {"ssh": "m", "ubaro": "m/s", "vbaro": "m/s"}
VOLUME = {"temp": "degC", "salin": "psu", "u": "m/s", "v": "m/s"}


@dataclass(frozen=True)
class Functional:
    """value(x) = sum(w * x[:, cols]) for a (grid, feature) state x."""

    name: str
    units: str
    cols: np.ndarray
    w: np.ndarray  # (grid, len(cols))


def functionals(names: list[str], area: np.ndarray, region: np.ndarray, level_ocean: np.ndarray, levels: np.ndarray) -> list[Functional]:
    col = {n: i for i, n in enumerate(names)}
    a = area * region
    out = [Functional(f"{v}_mean", u, np.array([col[v]]), (a / a.sum())[:, None]) for v, u in SURFACE.items()]
    vol = a[:, None] * level_weights(levels)[None] * level_ocean
    for var, units in VOLUME.items():
        cols = np.array([col[f"{var}_{d:g}m"] for d in levels])
        out.append(Functional(f"{var}_mean", units, cols, vol / vol.sum()))
        if var == "temp":
            out.append(Functional("heat_content", "J/m2", cols, RHO0 * CP * vol / a.sum()))
        if var == "salin":
            out.append(Functional("salt_content", "kg/m2", cols, RHO0 * 1e-3 * vol / a.sum()))
    return out


def columns(names: list[str], level_ocean: np.ndarray, levels: np.ndarray) -> list[Functional]:
    """Column heat and salt content at each point (J/m2, kg/m2): rho0 cp sum(T dz) and rho0 sum(S/1000 dz) over real levels."""
    col = {n: i for i, n in enumerate(names)}
    dz = level_weights(levels)[None] * level_ocean
    cols = {v: np.array([col[f"{v}_{d:g}m"] for d in levels]) for v in ("temp", "salin")}
    return [Functional("heat_column", "J/m2", cols["temp"], RHO0 * CP * dz), Functional("salt_column", "kg/m2", cols["salin"], RHO0 * 1e-3 * dz)]


def per_point(cs: list[Functional], x: np.ndarray) -> np.ndarray:
    """(..., grid, feature) -> (..., grid, len(cs))."""
    return np.stack([np.einsum("...gc,gc->...g", x[..., c.cols], c.w) for c in cs], axis=-1)


def apply(fs: list[Functional], x: np.ndarray) -> np.ndarray:
    """(..., grid, feature) -> (..., len(fs))."""
    return np.stack([np.einsum("...gc,gc->...", x[..., f.cols], f.w) for f in fs], axis=-1)


def offset_pattern(e: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a (..., grid) error into its w-weighted mean and the weighted RMSE of what remains: rmse^2 = offset^2 + pattern^2."""
    offset = np.einsum("...g,g->...", e, w) / w.sum()
    pattern = np.sqrt(np.einsum("...g,g->...", (e - offset[..., None]) ** 2, w) / w.sum())
    return offset, pattern


def summarize(true: np.ndarray, pred: np.ndarray, fs: list[Functional]) -> dict:
    """true, pred: (forecast, 1 + lead, quantity), index 0 the initial state. Per quantity and lead (from 1)."""
    err = pred - true
    step_t, step_p = np.diff(true, axis=1), np.diff(pred, axis=1)
    return {
        f.name: {
            "units": f.units,
            "leads": {
                str(lead): {
                    "error_mean": float(err[:, lead, q].mean()),
                    "error_std": float(err[:, lead, q].std()),
                    "positive_fraction": float((err[:, lead, q] > 0).mean()),
                    "truth_step": float(step_t[:, lead - 1, q].mean()),
                    "model_step": float(step_p[:, lead - 1, q].mean()),
                }
                for lead in range(1, err.shape[1])
            },
        }
        for q, f in enumerate(fs)
    }


def _setup(config: Path):
    import xarray as xr

    meta = xr.open_zarr(pack_path(config) / "meta.zarr", consolidated=True).load()
    names = [str(n) for n in meta.state_feature.values]
    area = cell_area(meta)
    levels = meta.level.values.astype(float)
    masks = regions(meta)
    fs = {r: functionals(names, area, m, meta.level_ocean.values.astype(bool), levels) for r, m in masks.items()}
    return meta, names, area, masks, fs, columns(names, meta.level_ocean.values.astype(bool), levels)


def forecasts(config: Path, ckpt: Path, split: str, ar_steps: int, limit: int | None) -> tuple[dict, dict]:
    from hycom_emulator.evaluate_b00 import forecast, load

    ds, data, module = load(config, ckpt, split, ar_steps)
    meta, names, area, masks, fs, cs = _setup(config)
    assert names == ds.get_vars_names("state"), "datastore and meta.zarr disagree on state features"
    ssh = names.index("ssh")
    idx = np.arange(len(data)) if limit is None else np.linspace(0, len(data) - 1, min(limit, len(data))).astype(int)
    n, nr = len(idx), len(masks)
    out = {
        "true": np.zeros((n, nr, ar_steps + 1, len(fs["gulf"]))),
        "pred": np.zeros((n, nr, ar_steps + 1, len(fs["gulf"]))),
        "ssh_offset": np.zeros((n, nr, ar_steps)),
        "ssh_pattern": np.zeros((n, nr, ar_steps)),
        "ssh_error_map": np.zeros((ar_steps, area.size)),
        "t0": np.zeros(n, "datetime64[ns]"),
        "loss_domain_bias": np.zeros((ar_steps, len(names))),
        "real_point_bias": np.zeros((ar_steps, len(names))),
        "column_mse": np.zeros((ar_steps, nr, len(cs))),
    }
    loss_domain = ~meta.boundary_mask.values.astype(bool)
    real = point_weights(names, np.ones(area.size), loss_domain, meta.level_ocean.values.astype(bool), meta.level.values.astype(float))
    diff_std = meta.state_diff_std.values
    for i, k in enumerate(idx):
        sample = data[k]
        x0, truth = sample[0][-1].numpy(), sample[1].numpy()
        pred = forecast(module, sample)
        out["t0"][i] = np.datetime64(int(sample[3][0]), "ns") - np.timedelta64(1, "D")
        states_t, states_p = np.concatenate([x0[None], truth]), np.concatenate([x0[None], pred])
        e = pred[..., ssh] - truth[..., ssh]
        out["ssh_error_map"] += e / n
        err = (pred - truth) / diff_std
        out["loss_domain_bias"] += err[:, loss_domain].mean(1) / n
        out["real_point_bias"] += np.einsum("lgf,gf->lf", err, real) / real.sum(0) / n
        ec = per_point(cs, pred) - per_point(cs, truth)
        for r, (name, mask) in enumerate(masks.items()):
            out["column_mse"][:, r] += np.einsum("lgc,g->lc", ec**2, area * mask) / (area * mask).sum() / n
            out["true"][i, r], out["pred"][i, r] = apply(fs[name], states_t), apply(fs[name], states_p)
            out["ssh_offset"][i, r], out["ssh_pattern"][i, r] = offset_pattern(e, area * mask)
    summary = {
        "label": "free forecast given the true daily wind and boundary band (no increments, no observations after t0)",
        "split": split, "ar_steps": ar_steps, "checkpoint": str(ckpt), "samples": n,
        "regions": {
            name: summarize(out["true"][:, r], out["pred"][:, r], fs[name])
            | {"ssh_split": {str(l + 1): {"offset_mean": float(out["ssh_offset"][:, r, l].mean()),
                                          "pattern_mean": float(out["ssh_pattern"][:, r, l].mean())} for l in range(ar_steps)}}
            for r, name in enumerate(masks)
        },
    }
    return summary, out | {"regions": np.array(list(masks)), "quantities": np.array([f.name for f in fs["gulf"]])}


def series(config: Path) -> dict:
    """Every quantity on every row of the pack: (time, region, quantity)."""
    meta, names, area, masks, fs, cs = _setup(config)
    train = meta.time.values < np.datetime64(meta.attrs["train_end"]) + np.timedelta64(1, "D")
    real = ~meta.time_filled.values
    change_ms, pairs, prev = np.zeros((len(masks), len(cs))), 0, None
    path = pack_path(config) / "state.npy"
    head = np.load(path, mmap_mode="r")
    shape, dtype, offset = head.shape, head.dtype, head.offset
    del head
    row = int(np.prod(shape[1:])) * dtype.itemsize
    values = np.zeros((shape[0], len(masks), len(fs["gulf"])))
    fd = os.open(path, os.O_RDONLY)
    try:
        for t in range(shape[0]):
            x = np.frombuffer(os.pread(fd, row, offset + t * row), dtype).reshape(shape[1:])
            os.posix_fadvise(fd, offset + t * row, row, os.POSIX_FADV_DONTNEED)
            for r, name in enumerate(masks):
                values[t, r] = apply(fs[name], x)
            now = per_point(cs, x) if train[t] and real[t] else None
            if now is not None and prev is not None:
                for r, mask in enumerate(masks.values()):
                    change_ms[r] += np.einsum("gc,g->c", (now - prev) ** 2, area * mask) / (area * mask).sum()
                pairs += 1
            prev = now
    finally:
        os.close(fd)
    return {"time": meta.time.values, "time_filled": meta.time_filled.values, "values": values,
            "column_change_ms": change_ms / pairs, "columns": np.array([c.name for c in cs]),
            "regions": np.array(list(masks)), "quantities": np.array([f.name for f in fs["gulf"]])}


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("forecast")
    f.add_argument("config", type=Path)
    f.add_argument("ckpt", type=Path)
    f.add_argument("out", type=Path, help="prefix of the .json summary and the .npz arrays")
    f.add_argument("--split", default="test")
    f.add_argument("--ar-steps", type=int, default=4)
    f.add_argument("--limit", type=int, default=None, help="score n forecasts spread evenly over the split")
    s = sub.add_parser("series")
    s.add_argument("config", type=Path)
    s.add_argument("out", type=Path, help="prefix of the .npz")
    a = p.parse_args()
    if a.command == "forecast":
        summary, arrays = forecasts(a.config, a.ckpt, a.split, a.ar_steps, a.limit)
        a.out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
        np.savez(a.out.with_suffix(".npz"), **arrays)
        for name, q in summary["regions"].items():
            for k in ("ssh_mean", "temp_mean", "salin_mean", "heat_content", "salt_content"):
                print(name, k, " ".join(f"{l}d {v['error_mean']:+.4g}" for l, v in q[k]["leads"].items()))
    else:
        np.savez(a.out.with_suffix(".npz"), **series(a.config))


if __name__ == "__main__":
    main()
