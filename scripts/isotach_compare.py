"""Hausdorff distance of the 1.5 kt isotach vs its surrogate loss, on persistence of a B00 pack.

For each lead L and each pair of days (t, t+L) inside a split, truth(t) is the forecast of
truth(t+L). Prints per-lead medians and the Spearman correlation of the surrogate with each
Hausdorff variant over the samples where both fronts exist; writes every sample to the JSON.

Run `python scripts/isotach_compare.py <pack_dir> <out.json> [--leads 1 2] [--max-samples N] [--lon-min -90]`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from scipy.stats import spearmanr

from hycom_emulator.isotach import (
    FrontStatus,
    front_distance,
    gulf_region,
    isotach_loss,
    isotach_target,
    mercator_cell_km,
    surface_speed,
    to_grid,
)

SPLITS = {"val": ("2025-08-06", "2025-08-15"), "test": ("2025-08-21", "2025-09-01")}  # scripts/skynet/train_b00.sh
METRICS = ("hausdorff_km", "mean_km", "p95_km")


class Pack:
    def __init__(self, path: Path, lon_min: float | None = None):
        meta = xr.open_zarr(path / "meta.zarr", consolidated=True)
        self.times = meta.time.values.astype("datetime64[D]")
        names = [str(f) for f in meta.state_feature.values]
        self.cols = [names.index(n) for n in ("u_k01", "v_k01", "ubaro", "vbaro")]
        x = meta.x.values
        nx = int(np.unique(x).size)
        self.shape = (x.size // nx, nx)
        static = {str(n): to_grid(meta.static.sel(static_feature=n).values, self.shape) for n in ("lon", "lat", "depth")}
        self.region = gulf_region(static["lon"], static["lat"], static["depth"])
        if lon_min is not None:
            self.region &= static["lon"] >= lon_min
        self.cell_km = mercator_cell_km(static["lat"])
        self.state = np.load(path / "state.npy", mmap_mode="r")
        self._speed: dict[int, np.ndarray] = {}

    def speed(self, i: int) -> np.ndarray:
        if i not in self._speed:
            a = np.asarray(self.state[i][:, self.cols], dtype=np.float64).T
            self._speed[i] = surface_speed(*to_grid(a, self.shape))
        return self._speed[i]

    def pairs(self, split: str, lead_days: int) -> list[tuple[int, int]]:
        lo, hi = (np.datetime64(d) for d in SPLITS[split])
        lead = np.timedelta64(lead_days, "D")
        idx = {t: i for i, t in enumerate(self.times)}
        return [(idx[t], idx[t + lead]) for t in self.times if lo <= t and t + lead <= hi and t + lead in idx]


def sample(pack: Pack, i0: int, i1: int, timings: dict[str, list[float]]) -> dict:
    pred, true = pack.speed(i0), pack.speed(i1)
    t = time.perf_counter()
    fd = front_distance(pred, true, pack.region, pack.cell_km)
    timings["front_distance"].append(time.perf_counter() - t)
    t = time.perf_counter()
    q, d_q = isotach_target(true, pack.region, pack.cell_km)
    timings["isotach_target"].append(time.perf_counter() - t)
    sp = torch.from_numpy(pred[None]).float().requires_grad_()
    t = time.perf_counter()
    loss = isotach_loss(sp, torch.from_numpy(q[None]).float(), torch.from_numpy(d_q[None]).float(), pack.region, pack.cell_km)
    timings["isotach_loss_fwd"].append(time.perf_counter() - t)
    t = time.perf_counter()
    loss.backward()
    timings["isotach_loss_bwd"].append(time.perf_counter() - t)
    return {"init": str(pack.times[i0]), "valid": str(pack.times[i1]), "status": str(fd.status),
            **{m: float(getattr(fd, m)) for m in METRICS}, "surrogate": loss.item()}


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == FrontStatus.OK]
    out = {"n": len(rows), "n_ok": len(ok)}
    for m in (*METRICS, "surrogate"):
        out[f"median_{m}"] = float(np.median([r[m] for r in ok])) if ok else float("nan")
    for m in METRICS:
        out[f"spearman_surrogate_{m}"] = float(spearmanr([r["surrogate"] for r in ok], [r[m] for r in ok]).statistic) if len(ok) > 2 else float("nan")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pack", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--leads", type=int, nargs="+", default=[1, 2])
    p.add_argument("--max-samples", type=int, default=None, help="per lead, for a smoke run")
    p.add_argument("--lon-min", type=float, default=None, help="also drop the region west of this longitude")
    a = p.parse_args()
    pack = Pack(a.pack, a.lon_min)
    timings: dict[str, list[float]] = {k: [] for k in ("front_distance", "isotach_target", "isotach_loss_fwd", "isotach_loss_bwd")}
    result = {"pack": str(a.pack), "splits": SPLITS, "lon_min": a.lon_min, "region_pixels": int(pack.region.sum()), "leads": {}}
    for lead in a.leads:
        pairs = [pr for split in SPLITS for pr in pack.pairs(split, lead)][: a.max_samples]
        rows = [sample(pack, i0, i1, timings) for i0, i1 in pairs]
        for r in rows:
            print(lead, r["init"], r["status"], *(f"{r[m]:.1f}" for m in METRICS), f"{r['surrogate']:.4g}", flush=True)
        result["leads"][lead] = {"summary": summarize(rows), "samples": rows}
        print(f"lead {lead}:", json.dumps(result["leads"][lead]["summary"]), flush=True)
    result["ms_per_call"] = {k: 1e3 * float(np.median(v)) for k, v in timings.items()}
    print("ms per call (median):", json.dumps(result["ms_per_call"]))
    a.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
