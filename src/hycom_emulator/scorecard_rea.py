"""Physicality scorecard of free runs: the lead at which each diagnostic leaves the truth.

    python -m hycom_emulator.scorecard_rea <stats.nc> <out_dir> [--test-from 2024-02-02]

Reads a `rollout_rea run` stats.nc. Each diagnostic is a model/truth ratio per (start, lead), or for the Gulf means a
model-minus-truth offset. Its departure lead, per start set, is the first lead where the median over starts (mean for
eddy counts) is outside the diagnostic's band and stays outside for the next two leads. The ties also report how big
the truth's own residual is against its lhs, over in-sample starts: a tie whose truth residual is over half its lhs is
marked unusable as a loss. Writes scorecard.csv, scorecard.md and scorecard.png.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from hycom_emulator import operators as op
from hycom_emulator.rollout_rea import BANDS, MEANS, SPECTRA, TIES

RATIO_BAND = (0.9, 1.1)
INVERSION_BAND = (0.5, 2.0)
COUNT_BAND = (0.67, 1.5)
FRONT_BAND = (0.0, 1.0)  # model front distance over persistence's: departs when the front is placed no better than persistence
RUN = 3  # leads outside the band in a row
TRUTH_RESIDUAL_MAX = 0.5
REPORT_LEADS = (1, 4, 7, 10, 14, 21, 30, 45, 60)


@dataclass(frozen=True)
class Diagnostic:
    name: str
    family: str  # blur, drift, eddies, ties, structure
    band: tuple[float, float]
    reduce: str  # median or mean over starts
    series: Callable[[xr.Dataset], xr.DataArray]


def ratio(model: str, truth: str | None = None) -> Callable[[xr.Dataset], xr.DataArray]:
    return lambda ds: ds[model] / ds[truth] if truth else ds[model]


def diagnostics(ds: xr.Dataset) -> list[Diagnostic]:
    d = [Diagnostic("ke", "blur", RATIO_BAND, "median", ratio("ke_model", "ke_truth")),
         Diagnostic("maxspeed", "blur", RATIO_BAND, "median", ratio("maxspeed_model", "maxspeed_truth"))]
    d += [Diagnostic(f"spec_{ch}_{b}", "blur", RATIO_BAND, "median", ratio(f"spec_{ch}_{b}_ratio")) for ch in SPECTRA for b in BANDS]
    for ch in MEANS:
        truth = ds[f"mean_{ch}_truth"]
        s = float((truth - truth.shift(lead=7)).std())  # the truth's own 7-day change of the Gulf mean
        d.append(Diagnostic(f"mean_{ch}", "drift", (-s, s), "median",
                            lambda x, ch=ch: x[f"mean_{ch}_model"] - x[f"mean_{ch}_truth"]))
    d += [Diagnostic("front_vs_persistence", "eddies", FRONT_BAND, "median", ratio("front_mean_km_model", "front_mean_km_pers")),
          Diagnostic("lc_area", "eddies", RATIO_BAND, "median", ratio("lc_area_model", "lc_area_truth")),
          Diagnostic("lc_lce", "eddies", COUNT_BAND, "mean", ratio("lc_lce_model", "lc_lce_truth"))]
    for sc in op.SCALES_KM:
        for k in [f"zeta_{z:g}m_{sc}" for z in op.VORTICITY_DEPTHS] + [f"div_0m_{sc}"]:
            d.append(Diagnostic(k, "blur", RATIO_BAND, "median", ratio(f"op_{k}_model", f"op_{k}_truth")))
        d += [Diagnostic(f"{t}_{sc}", "ties", RATIO_BAND, "median",
                         ratio(f"op_{t}_{sc}_residual_model", f"op_{t}_{sc}_residual_truth")) for t in TIES]
    d.append(Diagnostic("inversions", "structure", INVERSION_BAND, "median", ratio("op_inversions_model", "op_inversions_truth")))
    return d


def departure(curve: np.ndarray, band: tuple[float, float]) -> float:
    """First lead (1-based) where curve is outside band for RUN leads in a row; nan if never."""
    out = ~((curve >= band[0]) & (curve <= band[1])) & np.isfinite(curve)
    for i in range(len(out) - RUN + 1):
        if out[i:i + RUN].all():
            return float(i + 1)
    return np.nan


def reduce(x: xr.DataArray, how: str) -> np.ndarray:
    return (x.median("start") if how == "median" else x.mean("start")).values


def truth_residual_share(ds: xr.Dataset) -> dict[str, float]:
    ins = ds.sel(start=ds.in_sample)
    return {f"{t}_{sc}": float((ins[f"op_{t}_{sc}_residual_truth"] / ins[f"op_{t}_{sc}_lhs_truth"]).median())
            for sc in op.SCALES_KM for t in TIES}


def scorecard(ds: xr.Dataset, test_from: str) -> list[dict]:
    sets = {"test": ds.sel(start=ds.start >= np.datetime64(test_from)), "in_sample": ds.sel(start=ds.in_sample)}
    share = truth_residual_share(ds)
    rows = []
    for dg in diagnostics(ds):
        row = {"diagnostic": dg.name, "family": dg.family, "band": f"{dg.band[0]:.3g}..{dg.band[1]:.3g}"}
        for name, sub in sets.items():
            curve = reduce(dg.series(sub), dg.reduce)
            row[f"departs_{name}"] = departure(curve, dg.band)
            if name == "test":
                row |= {f"d{l}": float(curve[l - 1]) if l <= len(curve) else np.nan for l in REPORT_LEADS}
        row["n_test"] = int(sets["test"].sizes["start"])
        if dg.name in share:
            row["truth_residual_share"] = share[dg.name]
            row["usable_as_loss"] = share[dg.name] <= TRUTH_RESIDUAL_MAX
        rows.append(row)
    return sorted(rows, key=lambda r: (np.nan_to_num(r["departs_test"], nan=1e9), r["diagnostic"]))


COLUMNS = ["diagnostic", "family", "band", "departs_test", "departs_in_sample", *[f"d{l}" for l in REPORT_LEADS],
           "n_test", "truth_residual_share", "usable_as_loss"]


def _md(rows: list[dict], label: str) -> str:
    cols = COLUMNS
    fmt = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.3g}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = [f"# Physicality scorecard\n\n{label}\n", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def figure(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratios = [r for r in rows if r["family"] != "drift"]
    leads = list(REPORT_LEADS)
    v = np.log2(np.array([[r[f"d{l}"] for l in leads] for r in ratios], float))
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(ratios) + 1.2))
    im = ax.imshow(v, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(leads)), [str(l) for l in leads])
    ax.set_yticks(range(len(ratios)), [r["diagnostic"] for r in ratios], fontsize=7)
    ax.set_xlabel("lead (days)")
    fig.colorbar(im, ax=ax, label="log2(model / truth), test starts")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("stats", type=Path)
    a.add_argument("out", type=Path)
    a.add_argument("--test-from", default="2024-02-02")
    args = a.parse_args()
    ds = xr.open_dataset(args.stats)
    rows = scorecard(ds, args.test_from)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "scorecard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    label = f"stats: {args.stats}; checkpoint md5 {ds.attrs.get('checkpoint_md5', '?')}; test starts from {args.test_from}"
    (args.out / "scorecard.md").write_text(_md(rows, label))
    figure(rows, args.out / "scorecard.png")


if __name__ == "__main__":
    main()
