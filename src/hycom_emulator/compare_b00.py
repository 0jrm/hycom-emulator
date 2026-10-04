"""Compare several B00 checkpoints on the same forecast: one sample, one lead, every model's error side
by side with a shared colour scale per row.

Models are given as label=<nlam.yaml>:<checkpoint>[:<neural-lam model>] (the model loads the weights
into another step predictor; hycom_graph_lam applies the thickness projection). The sample is the best,
median or worst of the split by the first model's skill (mean over T, S, u, v, SSH of RMSE(model) /
RMSE(persistence + increments)), or a WeatherDataset index.

  compare_surface.png        target, then model - target for each model and persistence+inc - target:
                             T, S, total u, total v of the top layer, and SSH
  compare_ssh_change.png     the 24 h SSH change of the truth and of each model; grid-scale noise shows here
  compare_inversions.png     per column, the adjacent layer pairs (both thicker than 1 m) whose sigma2 is
                             lower below by more than 0.02 kg/m3, truth and each model
  compare_section_<s>.png    T, S and sigma2 error of each model on a depth section
  compare.json               the sample, and each model's skill on it

Run `python -m hycom_emulator.compare_b00 <out_dir> label=<nlam.yaml>:<ckpt>[:<model>] ... [--split test]
[--lead 1] [--sample median] [--lat 25] [--lon -86] [--depth 1500]`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hycom_emulator.evaluate_b00 import G, load
from hycom_emulator.inspect_b00 import Case, Forcing, Grid, Section, _limits
from hycom_emulator.physics import ONEM, layer_columns, sigma2

SURFACE = (("T k01 (degC)", "temp_k01", None), ("S k01 (psu)", "salin_k01", None),
           ("u k01 total (m/s)", "u_k01", "ubaro"), ("v k01 total (m/s)", "v_k01", "vbaro"), ("SSH (m)", "srfhgt", None))


def compare(out: Path, models: dict[str, tuple[Path, Path, str | None]], split="test", lead=1, sample="median",
            lats=(25.0,), lons=(-86.0,), depth=1500.0) -> dict:
    loaded = {label: load(cfg, ckpt, split, ar_steps=max(2, lead), model=model) for label, (cfg, ckpt, model) in models.items()}
    ds, data, _ = next(iter(loaded.values()))
    names = ds.get_vars_names("state")
    interior = ~ds.boundary_mask.values.astype(bool)
    area = np.cos(np.deg2rad(ds.get_dataarray("static", None).sel(static_feature="lat").values))[interior] ** 2
    forcing_of = Forcing(ds, split)
    first = next(iter(loaded.values()))[2]
    if sample in ("best", "median", "worst"):
        skills = [Case(ds, data, first, forcing_of, i, lead - 1).skill(names, interior, area) for i in range(len(data))]
        order = np.argsort(skills)
        idx = int(order[{"best": 0, "median": len(order) // 2, "worst": -1}[sample]])
    else:
        idx = int(sample)
    cases = {label: Case(d, dat, m, Forcing(d, split), idx, lead - 1) for label, (d, dat, m) in loaded.items()}
    ref = next(iter(cases.values()))
    title = f"{ref.title}, +{24 * lead} h (sample {idx}, {sample} by {next(iter(models))})"
    out.mkdir(parents=True, exist_ok=True)
    grid = Grid(ds)
    _surface(grid, names, cases, ref, title, out / "compare_surface.png")
    _ssh_change(grid, names, cases, ref, title, out / "compare_ssh_change.png")
    _inversions(grid, names, cases, ref, title, out / "compare_inversions.png")
    for axis, values in (("lat", lats), ("lon", lons)):
        for v in values:
            _section(grid, names, cases, ref, title, axis, v, depth, out / f"compare_section_{axis}{v:g}.png")
    res = {"split": split, "lead_h": 24 * lead, "sample": idx, "member": ref.member, "date": ref.date,
           "skill": {label: c.skill(names, interior, area) for label, c in cases.items()},
           "models": {label: [str(p) if p else None for p in spec] for label, spec in models.items()}}
    (out / "compare.json").write_text(json.dumps(res, indent=1))
    return res


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _field(names, x, var, baro):
    v = x[:, names.index(var)]
    if baro:
        v = v + x[:, names.index(baro)]
    return v / G if var == "srfhgt" else v


def _map(ax, grid, a, cmap, vmin, vmax, title):
    im = ax.pcolormesh(grid.lon, grid.lat, a, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
    ax.set_facecolor("0.85")
    ax.set_aspect(1 / np.cos(np.deg2rad(grid.lat.mean())))
    ax.set_title(title, fontsize=9)
    return im


def _surface(grid, names, cases, ref, title, path):
    plt = _plt()
    cols = 2 + len(cases)
    fig, axes = plt.subplots(len(SURFACE), cols, figsize=(4.2 * cols, 3.2 * len(SURFACE)), constrained_layout=True)
    for r, (label, var, baro) in enumerate(SURFACE):
        true = grid.map(_field(names, ref.true, var, baro))
        errs = {k: grid.map(_field(names, c.model, var, baro)) - true for k, c in cases.items()}
        errs["persistence+inc"] = grid.map(_field(names, ref.pinc, var, baro)) - true
        lo, hi = _limits(true)
        dlo, dhi = _limits(*errs.values(), symmetric=True)
        fig.colorbar(_map(axes[r, 0], grid, true, "viridis", lo, hi, f"{label}: target"), ax=axes[r, 0], shrink=0.8)
        for ax, (k, e) in zip(axes[r, 1:], errs.items()):
            rms = float(np.sqrt(np.nanmean(e**2)))
            im = _map(ax, grid, e, "RdBu_r", dlo, dhi, f"{k} - target (rms {rms:.3g})")
        fig.colorbar(im, ax=axes[r, -1], shrink=0.8)
    fig.suptitle(title)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _ssh_change(grid, names, cases, ref, title, path):
    plt = _plt()
    col = names.index("srfhgt")
    chg = {"truth": grid.map((ref.true[:, col] - ref.x0[:, col]) / G)}
    chg |= {k: grid.map((c.model[:, col] - c.x0[:, col]) / G) for k, c in cases.items()}
    lo, hi = _limits(*chg.values(), symmetric=True)
    fig, axes = plt.subplots(1, len(chg), figsize=(4.2 * len(chg), 3.6), constrained_layout=True)
    for ax, (k, a) in zip(axes, chg.items()):
        lap = a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1]
        im = _map(ax, grid, a, "RdBu_r", lo, hi, f"{k}: 24 h SSH change (m), Laplacian rms {np.sqrt(np.nanmean(lap**2)):.2e}")
    fig.colorbar(im, ax=axes[-1], shrink=0.8)
    fig.suptitle(title)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _inversions(grid, names, cases, ref, title, path):
    plt = _plt()
    th, tc, sc = (layer_columns(names, v) for v in ("thknss", "temp", "salin"))

    def count(x):
        sig, dp = sigma2(x[:, tc], x[:, sc]), x[:, th] / ONEM
        bad = (sig[:, 1:] < sig[:, :-1] - 0.02) & (dp[:, 1:] > 1) & (dp[:, :-1] > 1)
        return grid.map(bad.sum(1).astype(float))

    maps = {"truth": count(ref.true)} | {k: count(c.model) for k, c in cases.items()}
    fig, axes = plt.subplots(1, len(maps), figsize=(4.2 * len(maps), 3.6), constrained_layout=True)
    for ax, (k, a) in zip(axes, maps.items()):
        im = _map(ax, grid, np.where(a > 0, a, np.nan), "magma_r", 1, 6, f"{k}: inverted layer pairs, {int(np.nansum(a))} in all")
    fig.colorbar(im, ax=axes[-1], shrink=0.8, label="pairs per column")
    fig.suptitle(title)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _section(grid, names, cases, ref, title, axis, value, depth, path):
    plt = _plt()
    sec = Section(grid, names, ref.true, axis, value)
    tc, sc = (layer_columns(names, v) for v in ("temp", "salin"))
    rho = lambda x: np.where(sec.blank[:, None], np.nan, sigma2(x[sec.gi][:, tc], x[sec.gi][:, sc]))  # noqa: E731
    fields = {"T error (degC)": lambda x: sec.values(x, "temp") - sec.values(ref.true, "temp"),
              "S error (psu)": lambda x: sec.values(x, "salin") - sec.values(ref.true, "salin"),
              "sigma2 error (kg/m3)": lambda x: rho(x) - rho(ref.true)}
    errs = {f: {k: fn(c.model) for k, c in cases.items()} for f, fn in fields.items()}
    fig, axes = plt.subplots(len(cases), len(fields), figsize=(6 * len(fields), 2.8 * len(cases)), constrained_layout=True, squeeze=False)
    for j, (f, by_model) in enumerate(errs.items()):
        lo, hi = _limits(*by_model.values(), symmetric=True)
        for i, (k, e) in enumerate(by_model.items()):
            ax = axes[i, j]
            im = ax.pcolormesh(sec.X.T, sec.ze.T, e.T, cmap="RdBu_r", vmin=lo, vmax=hi, shading="flat")
            ax.set_ylim(depth, 0)
            ax.set_facecolor("0.85")
            ax.set_title(f"{k}: {f}", fontsize=9)
            ax.set_ylabel("depth (m)")
        fig.colorbar(im, ax=axes[:, j], shrink=0.6)
    fig.suptitle(f"{title}; section at {sec.where()}")
    fig.savefig(path, dpi=80)
    plt.close(fig)


def parse_model(spec: str) -> tuple[str, tuple[Path, Path, str | None]]:
    label, rest = spec.split("=", 1)
    parts = rest.split(":")
    return label, (Path(parts[0]), Path(parts[1]), parts[2] if len(parts) > 2 else None)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("models", nargs="+", help="label=<nlam.yaml>:<checkpoint>[:<neural-lam model>], first ranks the samples")
    p.add_argument("--split", default="test")
    p.add_argument("--lead", type=int, default=1)
    p.add_argument("--sample", default="median", help="best, median, worst or a WeatherDataset index")
    p.add_argument("--lat", type=float, nargs="*", default=[25.0])
    p.add_argument("--lon", type=float, nargs="*", default=[-86.0])
    p.add_argument("--depth", type=float, default=1500.0)
    a = p.parse_args()
    res = compare(a.out, dict(parse_model(m) for m in a.models), a.split, a.lead, a.sample, tuple(a.lat), tuple(a.lon), a.depth)
    print(json.dumps({k: res[k] for k in ("sample", "member", "date", "skill")}, indent=1))


if __name__ == "__main__":
    main()
