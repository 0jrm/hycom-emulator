"""Score a reanalysis-emulator checkpoint per field and lead against persistence, over the Gulf of Mexico.

    python -m hycom_emulator.evaluate_rea <nlam.yaml> <checkpoint> <out.json> [--split test] [--ar-steps 4]

Every state channel is scored at every lead: RMSE and bias of the model and of persistence, and corr_change, the
correlation of the predicted with the true change since the initial state. T, S, u and v are also scored as
column aggregates to 2000 m, each level weighted by the depth interval it represents. Points: the Gulf (static
`gulf`; or, with --region interior, all ocean as evaluate_b00 scores it), outside the boundary band, real at that level (`level_ocean`); weight cos^2(lat), the cell area of the
Mercator grid. These are free forecasts given the true daily wind and the true boundary band: the pack has no
increment channels and no observation enters after t0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

LEVEL_VARS = ("temp", "salin", "u", "v")
SUMS = ("w", "e2_model", "e2_pers", "e_model", "e_pers", "a", "b", "aa", "bb", "ab")


def level_weights(levels: np.ndarray, bottom: float = 2000.0) -> np.ndarray:
    """Depth interval (m) each level stands for: from the midpoint above to the midpoint below, 0 to `bottom`."""
    edges = np.concatenate([[0.0], (levels[1:] + levels[:-1]) / 2, [bottom]])
    return np.diff(edges)


def point_weights(names: list[str], area: np.ndarray, region: np.ndarray, level_ocean: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """(grid, feature) weights: area on region points that are real for that channel, else 0."""
    w = np.zeros((area.size, len(names)))
    for j, n in enumerate(names):
        var, _, depth = n.rpartition("_")
        real = level_ocean[:, int(np.argmin(np.abs(levels - float(depth[:-1]))))] if var in LEVEL_VARS else True
        w[:, j] = area * (region & real)
    return w


def accumulate(acc: dict, x0: np.ndarray, truth: np.ndarray, pred: np.ndarray, w: np.ndarray) -> None:
    """Add one forecast: x0 (grid, feature), truth and pred (lead, grid, feature), w (grid, feature)."""
    em, ep = pred - truth, x0[None] - truth
    a, b = pred - x0[None], truth - x0[None]
    terms = {"e2_model": em**2, "e2_pers": ep**2, "e_model": em, "e_pers": ep, "a": a, "b": b, "aa": a * a, "bb": b * b, "ab": a * b}
    acc["w"] = acc.get("w", 0.0) + np.repeat(w.sum(0)[None], em.shape[0], 0)
    for k, v in terms.items():
        acc[k] = acc.get(k, 0.0) + np.einsum("lgf,gf->lf", v, w)


def summarize(acc: dict, names: list[str], dz: np.ndarray) -> dict:
    """Per channel and per column aggregate: {name: {lead: scores}} from the accumulated weighted sums."""
    groups = {n: [(j, 1.0)] for j, n in enumerate(names)}
    for var in LEVEL_VARS:
        groups[var] = [(j, dz[k]) for k, j in enumerate(j for j, n in enumerate(names) if n.rpartition("_")[0] == var)]
    out = {}
    for name, members in groups.items():
        s = {k: sum(c * acc[k][:, j] for j, c in members) for k in SUMS}
        cov = s["ab"] - s["a"] * s["b"] / s["w"]
        den = np.sqrt((s["aa"] - s["a"] ** 2 / s["w"]) * (s["bb"] - s["b"] ** 2 / s["w"]))
        out[name] = {
            str(lead + 1): {
                "rmse_model": float(np.sqrt(s["e2_model"][lead] / s["w"][lead])),
                "rmse_persistence": float(np.sqrt(s["e2_pers"][lead] / s["w"][lead])),
                "bias_model": float(s["e_model"][lead] / s["w"][lead]),
                "corr_change": float(cov[lead] / den[lead]) if den[lead] > 0 else float("nan"),
            }
            for lead in range(len(s["w"]))
        }
    return out


def pack_path(config: Path) -> Path:
    """The rea_pack folder a neural-lam config's datastore reads."""
    return Path(yaml.safe_load((config.parent / yaml.safe_load(config.read_text())["datastore"]["config_path"]).read_text())["zarr"])


def regions(meta: xr.Dataset) -> dict[str, np.ndarray]:
    """Scored points per region: interior is ocean outside the boundary band, gulf the interior inside the static gulf mask."""
    static = meta.static
    interior = ~meta.boundary_mask.values.astype(bool) & static.sel(static_feature="ocean").values.astype(bool)
    return {"gulf": interior & static.sel(static_feature="gulf").values.astype(bool), "interior": interior}


def cell_area(meta: xr.Dataset) -> np.ndarray:
    """cos^2(lat): the Mercator cell area up to a constant."""
    return np.cos(np.deg2rad(meta.static.sel(static_feature="lat").values)) ** 2


def in_period(target_times_ns: np.ndarray, period: tuple[str, str] | None) -> bool:
    """True if the two initial days (the two days before the first target) and every target day lie in period."""
    if period is None:
        return True
    t = np.asarray(target_times_ns).astype("datetime64[ns]")
    first_init = t[0] - np.timedelta64(2, "D")
    return bool(first_init >= np.datetime64(period[0]) and t[-1] <= np.datetime64(period[1]) + np.timedelta64(1, "D") - np.timedelta64(1, "ns"))


def evaluate(config: Path, ckpt: Path, split: str, ar_steps: int, limit: int | None = None, region_name: str = "gulf",
             period: tuple[str, str] | None = None) -> dict:
    from hycom_emulator.evaluate_b00 import forecast, load

    ds, data, module = load(config, ckpt, split, ar_steps)
    meta = xr.open_zarr(pack_path(config) / "meta.zarr", consolidated=True).load()
    names = [str(n) for n in meta.state_feature.values]
    assert names == ds.get_vars_names("state"), "datastore and meta.zarr disagree on state features"
    levels = meta.level.values.astype(float)
    w = point_weights(names, cell_area(meta), regions(meta)[region_name], meta.level_ocean.values.astype(bool), levels)
    acc: dict = {}
    indices = [i for i in range(len(data)) if in_period(data[i][3].numpy(), period)][:limit]
    for i in indices:
        sample = data[i]
        accumulate(acc, sample[0][-1].numpy(), sample[1].numpy(), forecast(module, sample), w)
    return {
        "label": "free forecast given the true daily wind and boundary band (no increments, no observations after t0)",
        "region": region_name, "period": list(period) if period else None, "split": split, "ar_steps": ar_steps, "checkpoint": str(ckpt), "samples": len(indices),
        "fields": summarize(acc, names, level_weights(levels)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("ckpt", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--split", default="test")
    p.add_argument("--ar-steps", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="score only the first n samples (smoke)")
    p.add_argument("--region", choices=("gulf", "interior"), default="gulf", help="interior: every ocean point outside the boundary band, as evaluate_b00 scores")
    p.add_argument("--period", nargs=2, metavar=("START", "END"), help="score only forecasts whose initial and target days lie in [START, END], e.g. one source experiment")
    a = p.parse_args()
    r = evaluate(a.config, a.ckpt, a.split, a.ar_steps, a.limit, a.region, tuple(a.period) if a.period else None)
    a.out.write_text(json.dumps(r, indent=1))
    for name in (*LEVEL_VARS, "ssh"):
        row = " ".join(f"{lead}d {s['rmse_model'] / s['rmse_persistence']:.3f}" for lead, s in r["fields"][name].items())
        print(f"{name:6s} RMSE/persistence {row}")


if __name__ == "__main__":
    main()
