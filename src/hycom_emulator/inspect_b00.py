"""Look at single B00 forecasts: skill per sample, then maps and depth sections of chosen days.

Skill of a sample at the chosen lead = mean over T, S, u, v, SSH of RMSE(model) / RMSE(persistence +
increments), weighted as in evaluate_b00 (below 1 beats the baseline). All samples of the split are
ranked, pooled over ensemble members, and the best, median and worst are drawn:

  samples.json                 skill of every sample, best first
  surface_<rank>.png           top-layer target, model, model - target, persistence+inc - target for every
                               state variable (thknss_k01 only if it varies by over 1 mm)
  section_<rank>_<lat|lon>.png target, model - target and persistence+inc - target of T, S, u, v on a
                               section, layers drawn between the target's interface depths

Boundary points (land and the nest band) are blank: neural-lam overwrites them with the truth.

Run `python -m hycom_emulator.inspect_b00 <nlam.yaml> <ckpt> <out_dir> [--split test] [--lead 1]
[--lat 25 27] [--lon -90 -86] [--depth 1500]`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hycom_emulator.evaluate_b00 import (
    FIELDS,
    G,
    field_values,
    forecast,
    increment_columns,
    load,
    persistence_inc,
    rmse,
    weights,
)

ONEM = 9806.0  # Pa of layer thickness per metre
SECTION_FIELDS = (("temp", "T (degC)"), ("salin", "S (psu)"), ("u", "u total (m/s)"), ("v", "v total (m/s)"))


class Case:
    """One sample at one lead: initial state, target, model and persistence+inc, (grid, state_feature)."""

    def __init__(self, ds, data, module, forcing_of, idx, lead):
        sample = data[idx]
        init, target, _, times = sample
        self.member = forcing_of.member(idx)
        self.date = str(np.datetime64(int(times[lead]), "ns"))[:10]
        self.x0 = init[-1].numpy()
        self.true = target[lead].numpy()
        self.model = forecast(module, sample)[lead]
        self.pinc = persistence_inc(self.x0, forcing_of(idx), times, increment_columns(ds))[lead]
        self.title = f"{self.member or ''} target {self.date}".strip()

    def skill(self, names, interior, area) -> float:
        fv = {k: field_values(getattr(self, k)[interior], names) for k in ("true", "model", "pinc")}
        w = weights(self.true[interior], names, area)
        return float(np.mean([rmse(fv["model"][f], fv["true"][f], w[f]) / rmse(fv["pinc"][f], fv["true"][f], w[f]) for f in FIELDS]))


class Forcing:
    """The split's forcing, per ensemble member when there are members."""

    def __init__(self, ds, split):
        self.da = ds.get_dataarray("forcing", split)
        self.members = self.da["ensemble_member"].values.tolist() if ds.is_ensemble else [None]

    def member(self, idx):
        return self.members[idx % len(self.members)]

    def __call__(self, idx):
        m = self.member(idx)
        return self.da if m is None else self.da.sel(ensemble_member=m)


def inspect(config: Path, ckpt: Path, out: Path, split="test", lead=1, lats=(25.0, 27.0), lons=(-90.0, -86.0), depth=1500.0):
    ds, data, module = load(config, ckpt, split, ar_steps=max(2, lead))
    names = ds.get_vars_names("state")
    interior = ~ds.boundary_mask.values.astype(bool)
    lat = ds.get_dataarray("static", None).sel(static_feature="lat").values
    area = np.cos(np.deg2rad(lat))[interior] ** 2
    forcing_of = Forcing(ds, split)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in range(len(data)):
        c = Case(ds, data, module, forcing_of, idx, lead - 1)
        rows.append({"idx": idx, "member": c.member, "date": c.date, "skill": c.skill(names, interior, area)})
    rows.sort(key=lambda r: r["skill"])
    (out / "samples.json").write_text(json.dumps({"lead_h": 24 * lead, "split": split, "checkpoint": str(ckpt), "samples": rows}, indent=1))

    grid = Grid(ds)
    for rank, row in (("best", rows[0]), ("median", rows[len(rows) // 2]), ("worst", rows[-1])):
        c = Case(ds, data, module, forcing_of, row["idx"], lead - 1)
        title = f"{rank} ({row['skill']:.2f} x persistence+inc RMSE): {c.title}, +{24 * lead} h"
        plot_surface(grid, names, c, title, out / f"surface_{rank}.png")
        for v in lats:
            plot_section(grid, names, c, title, "lat", v, depth, out / f"section_{rank}_lat{v:g}.png")
        for v in lons:
            plot_section(grid, names, c, title, "lon", v, depth, out / f"section_{rank}_lon{v:g}.png")
    return rows


class Grid:
    """Map between neural-lam's grid_index (x outer, y inner) and (y, x) arrays."""

    def __init__(self, ds):
        xy = ds.get_xy("state", stacked=False)
        self.lon, self.lat = xy[:, 0, 0], xy[0, :, 1]
        self.nx, self.ny = self.lon.size, self.lat.size
        self.blank = ds.boundary_mask.values.astype(bool)

    def map(self, v):
        return np.where(self.blank, np.nan, v).reshape(self.nx, self.ny).T

    def section(self, axis, value):
        """grid_index of the points along a section, and their coordinate along it."""
        if axis == "lat":
            j = int(np.argmin(np.abs(self.lat - value)))
            return np.arange(self.nx) * self.ny + j, self.lon, float(self.lat[j])
        i = int(np.argmin(np.abs(self.lon - value)))
        return i * self.ny + np.arange(self.ny), self.lat, float(self.lon[i])


def _surface_vars(names, true):
    col = {n: i for i, n in enumerate(names)}
    out = [
        ("T k01 (degC)", lambda x: x[:, col["temp_k01"]]),
        ("S k01 (psu)", lambda x: x[:, col["salin_k01"]]),
        ("u k01 total (m/s)", lambda x: x[:, col["u_k01"]] + x[:, col["ubaro"]]),
        ("v k01 total (m/s)", lambda x: x[:, col["v_k01"]] + x[:, col["vbaro"]]),
        ("SSH (m)", lambda x: x[:, col["srfhgt"]] / G),
        ("montg1 (m)", lambda x: x[:, col["montg1"]] / G),
        ("u barotropic (m/s)", lambda x: x[:, col["ubaro"]]),
        ("v barotropic (m/s)", lambda x: x[:, col["vbaro"]]),
    ]
    if np.ptp(true[:, col["thknss_k01"]]) > 1e-3 * ONEM:  # 05.3 and the twins hold it at 1 m (float noise 1e-7 m)
        out.append(("thknss k01 (m)", lambda x: x[:, col["thknss_k01"]] / ONEM))
    return out


def _limits(*arrays, pct=99.0, symmetric=False):
    v = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    if symmetric:
        m = float(np.percentile(np.abs(v), pct)) or 1.0
        return -m, m
    return float(np.percentile(v, 100 - pct)), float(np.percentile(v, pct))


def plot_surface(grid, names, c, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _surface_vars(names, c.true)
    fig, axes = plt.subplots(len(rows), 4, figsize=(19, 3.4 * len(rows)), constrained_layout=True)
    for r, (label, f) in enumerate(rows):
        true, model, pinc = (grid.map(f(x)) for x in (c.true, c.model, c.pinc))
        lo, hi = _limits(true)
        dlo, dhi = _limits(model - true, pinc - true, symmetric=True)
        panels = [(true, "target", "viridis", lo, hi), (model, "model", "viridis", lo, hi),
                  (model - true, "model - target", "RdBu_r", dlo, dhi), (pinc - true, "persistence+inc - target", "RdBu_r", dlo, dhi)]
        for ax, (a, name, cmap, vmin, vmax) in zip(axes[r], panels):
            im = ax.pcolormesh(grid.lon, grid.lat, a, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
            ax.set_facecolor("0.85")
            ax.set_aspect(1 / np.cos(np.deg2rad(grid.lat.mean())))
            ax.set_title(f"{label}: {name}", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _layers(names, var):
    return np.array(sorted((i for i, n in enumerate(names) if n.startswith(f"{var}_k")), key=lambda i: names[i]))


def plot_section(grid, names, c, title, axis, value, depth, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gi, coord, at = grid.section(axis, value)
    blank = grid.blank[gi]
    col = {n: i for i, n in enumerate(names)}
    z = np.concatenate([np.zeros((gi.size, 1)), np.cumsum(c.true[gi][:, _layers(names, "thknss")] / ONEM, axis=1)], axis=1)
    edges = np.concatenate([[coord[0] - (coord[1] - coord[0]) / 2], (coord[1:] + coord[:-1]) / 2, [coord[-1] + (coord[-1] - coord[-2]) / 2]])
    ze = np.concatenate([z[:1], (z[1:] + z[:-1]) / 2, z[-1:]])  # interface depths at the column edges
    X = np.broadcast_to(edges[:, None], ze.shape)

    def values(x, var):
        a = x[gi][:, _layers(names, var)]
        if var in ("u", "v"):
            a = a + x[gi][:, [col[f"{var}baro"]]]
        return np.where(blank[:, None], np.nan, a)

    fig, axes = plt.subplots(len(SECTION_FIELDS), 3, figsize=(18, 3.2 * len(SECTION_FIELDS)), constrained_layout=True)
    for r, (var, label) in enumerate(SECTION_FIELDS):
        true, model, pinc = (values(x, var) for x in (c.true, c.model, c.pinc))
        lo, hi = _limits(true)
        dlo, dhi = _limits(model - true, pinc - true, symmetric=True)
        panels = [(true, "target", "viridis", lo, hi), (model - true, "model - target", "RdBu_r", dlo, dhi),
                  (pinc - true, "persistence+inc - target", "RdBu_r", dlo, dhi)]
        for ax, (a, name, cmap, vmin, vmax) in zip(axes[r], panels):
            im = ax.pcolormesh(X.T, ze.T, a.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
            ax.set_ylim(depth, 0)
            ax.set_facecolor("0.85")
            ax.set_title(f"{label}: {name}", fontsize=9)
            ax.set_xlabel("longitude" if axis == "lat" else "latitude")
            ax.set_ylabel("depth (m)")
            fig.colorbar(im, ax=ax, shrink=0.8)
    where = f"{at:.2f}N" if axis == "lat" else f"{at:.2f}E"
    fig.suptitle(f"{title}; section at {where}")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("ckpt", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--split", default="test")
    p.add_argument("--lead", type=int, default=1, help="lead in days")
    p.add_argument("--lat", type=float, nargs="*", default=[25.0, 27.0])
    p.add_argument("--lon", type=float, nargs="*", default=[-90.0, -86.0])
    p.add_argument("--depth", type=float, default=1500.0, help="deepest metre shown on sections")
    a = p.parse_args()
    rows = inspect(a.config, a.ckpt, a.out, a.split, a.lead, tuple(a.lat), tuple(a.lon), a.depth)
    for r in rows:
        print(f"{r['skill']:.3f} {r['member'] or ''} {r['date']}")


if __name__ == "__main__":
    main()
