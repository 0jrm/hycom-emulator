"""Local linear operators on the reanalysis pack grid: the Gaussian case of a CRAM-style energy.

CRAM (Hoover, Blayney and Whittington, arXiv 2611.17309) learns a pairwise energy whose weights are shared across
positions, trains it with a local objective (each cell's conditional given the true values of its neighbours, no
solver in the loop), and puts the search into inference. On a continuous field the pairwise energy is Gaussian and each
local conditional is a linear regression on a window of neighbours. This module holds the machinery the cards built on
that idea share:

- window features: every state channel on a k x k window around a cell (weights shared by all cells), plus the
  forcing channels at the cell and a constant;
- classes: a cell's class is its number of real levels. Masks nest with depth (level_ocean is monotone), so a target
  on level l uses the cells of every class above l, and one Gram per class serves all 91 targets;
- GramSet: per class, sum phi phi^T, sum phi y^T and sum y^2 in float64, plus per-cell sums for a per-cell bias;
- ridge solves on cumulative Grams, and the held-out mean squared error of any weights in closed form from a
  held-out GramSet;
- StencilStep: dx(i) = W phi(x_t)(i) [+ b(i)], a one-day step that the rollout code runs like a neural-lam checkpoint
  (load_module, called by evaluate_b00.load for a .pt file).

Grid layout: grid_index = ix * ny + iy, images are field.reshape(nx, ny).T, as rollout_rea.Plane reads them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

KIND = "stencil"


def images(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """(B, grid, C) -> (B, C, ny, nx)."""
    nx, ny = shape
    return x.reshape(x.shape[0], nx, ny, x.shape[-1]).permute(0, 3, 2, 1)


def columns(img: torch.Tensor) -> torch.Tensor:
    """(B, C, ny, nx) -> (B, grid, C)."""
    b, c, ny, nx = img.shape
    return img.permute(0, 3, 2, 1).reshape(b, nx * ny, c)


def window(x: torch.Tensor, shape: tuple[int, int], k: int) -> torch.Tensor:
    """(B, grid, C) -> (B, grid, C * k * k): channel-major, then the window row-major; zero outside the image."""
    nx, ny = shape
    f = F.unfold(images(x, shape), k, padding=k // 2)  # (B, C k k, ny nx), positions row-major over (iy, ix)
    return f.reshape(f.shape[0], f.shape[1], ny, nx).permute(0, 3, 2, 1).reshape(f.shape[0], nx * ny, f.shape[1])


def centre(c: int, k: int) -> int:
    """Index of channel c at the window centre in window()'s feature order."""
    return c * k * k + (k * k) // 2


def features(x: torch.Tensor, forcing: torch.Tensor | None, shape: tuple[int, int], k: int) -> torch.Tensor:
    """(B, grid, P): window(x), the forcing at the cell, and a constant 1 last."""
    parts = [window(x, shape, k)]
    if forcing is not None:
        parts.append(forcing)
    parts.append(torch.ones(*x.shape[:2], 1, dtype=x.dtype, device=x.device))
    return torch.cat(parts, -1)


@dataclass(frozen=True)
class Layout:
    """Which cells and channels count. levels[c] is the channel's level index (0 for surface channels); a channel is
    real at cell i when nreal[i] > levels[c]. edge drops cells whose window leaves the image."""
    shape: tuple[int, int]  # (nx, ny)
    nreal: np.ndarray  # (grid,) int: number of real levels
    levels: np.ndarray  # (C,) int
    edge: np.ndarray  # (grid,) bool: within k // 2 of the image edge

    @property
    def nclass(self) -> int:
        return int(self.nreal.max())

    def valid(self) -> np.ndarray:
        """(grid, C) bool: channel c is real at the cell."""
        return self.nreal[:, None] > self.levels[None, :]

    @staticmethod
    def build(level_ocean: np.ndarray, names: list[str], depths: list[float], shape: tuple[int, int], k: int) -> "Layout":
        lo = np.asarray(level_ocean, bool)
        nreal = lo.sum(1)
        if not (lo == (np.arange(lo.shape[1])[None] < nreal[:, None])).all():
            raise ValueError("level_ocean is not monotone with depth; classes would not nest")
        lev = []
        for n in names:
            head, _, tail = n.rpartition("_")
            lev.append(depths.index(float(tail[:-1])) if head and tail.endswith("m") and tail[:-1].replace(".", "").isdigit() else 0)
        nx, ny = shape
        ix, iy = np.divmod(np.arange(nx * ny), ny)
        r = k // 2
        edge = (ix < r) | (ix >= nx - r) | (iy < r) | (iy >= ny - r)
        return Layout(shape, nreal, np.array(lev), edge)


@dataclass
class GramSet:
    """Per class g (cells with g real levels, g = 1..L): G[g] = sum phi phi^T, R[g] = sum phi y^T, Y2[g] = sum y^2,
    n[g] = rows. With per_cell, also per-cell sums of phi and y for the per-cell bias (CRAM's B_i)."""
    P: int
    C: int
    L: int
    G: torch.Tensor = None
    R: torch.Tensor = None
    Y2: torch.Tensor = None
    n: torch.Tensor = None
    days: int = 0
    per_cell: bool = False
    S_phi: torch.Tensor | None = None  # (grid, P)
    S_y: torch.Tensor | None = None  # (grid, C)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64)  # noqa: E731
        if self.G is None:
            self.G, self.R, self.Y2, self.n = z(self.L + 1, self.P, self.P), z(self.L + 1, self.P, self.C), z(self.L + 1, self.C), z(self.L + 1)

    def to(self, device) -> "GramSet":
        for a in ("G", "R", "Y2", "n", "S_phi", "S_y"):
            if getattr(self, a) is not None:
                setattr(self, a, getattr(self, a).to(device))
        return self

    def add(self, phi: torch.Tensor, y: torch.Tensor, rows_by_class: list[torch.Tensor | None], cells: torch.Tensor | None = None):
        """phi (grid, P) and y (grid, C) of one day. rows_by_class[g] are the cell rows of class g used for fitting."""
        if self.per_cell:
            if self.S_phi is None:
                self.S_phi = torch.zeros(phi.shape[0], self.P, dtype=torch.float64, device=phi.device)
                self.S_y = torch.zeros(phi.shape[0], self.C, dtype=torch.float64, device=phi.device)
            self.S_phi += phi.double()
            self.S_y += y.double()
        for g, rows in enumerate(rows_by_class):
            if rows is None or rows.numel() == 0:
                continue
            p, t = phi[rows].double(), y[rows].double()  # float32 Grams lose positive definiteness at ~1e-3
            self.G[g] += p.T @ p
            self.R[g] += p.T @ t
            self.Y2[g] += (t ** 2).sum(0)
            self.n[g] += rows.numel()
        self.days += 1

    def within(self, rows_by_class) -> "GramSet":
        """The fixed-effects Grams: each cell's mean over days removed (exact, from the per-cell sums)."""
        assert self.per_cell and self.days > 0
        out = GramSet(self.P, self.C, self.L, self.G.clone(), self.R.clone(), self.Y2.clone(), self.n.clone(), self.days)
        d = float(self.days)
        for g, rows in enumerate(rows_by_class):
            if rows is None or rows.numel() == 0:
                continue
            mp, my = self.S_phi[rows] / d, self.S_y[rows] / d
            out.G[g] -= d * mp.T @ mp
            out.R[g] -= d * mp.T @ my
            out.Y2[g] -= d * (my ** 2).sum(0)
        return out

    def cumulative(self, level: int):
        """(G, R, Y2, n) summed over classes g > level."""
        s = slice(level + 1, None)
        return self.G[s].sum(0), self.R[s].sum(0), self.Y2[s].sum(0), self.n[s].sum()

    def save(self, path: Path) -> None:
        torch.save({k: getattr(self, k) for k in ("P", "C", "L", "G", "R", "Y2", "n", "days", "per_cell", "S_phi", "S_y", "meta")}, path)

    @staticmethod
    def load(path: Path) -> "GramSet":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return GramSet(**d)


def rows_by_class(layout: Layout, keep: np.ndarray | None = None, device="cpu") -> list[torch.Tensor | None]:
    """Cell rows per class g = 0..L (class 0, land, is None), excluding edge cells and cells not in keep."""
    ok = ~layout.edge if keep is None else (~layout.edge & keep)
    out: list[torch.Tensor | None] = [None]
    for g in range(1, layout.nclass + 1):
        out.append(torch.as_tensor(np.flatnonzero(ok & (layout.nreal == g)), device=device))
    return out


def ridge(gs: GramSet, levels: np.ndarray, lam: float, drop: dict[int, int] | None = None, free_const: bool = True) -> torch.Tensor:
    """(C, P) weights. Channel c solves on the classes above its level: (G / n + lam I) w = R / n; the constant (last
    feature) is not penalised. drop[c] = a feature index excluded for channel c (its weight is 0)."""
    W = torch.zeros(gs.C, gs.P, dtype=torch.float64, device=gs.G.device)
    for lev in np.unique(levels):
        G, R, _, n = gs.cumulative(int(lev))
        if float(n) == 0:
            continue
        A = G / n
        pen = torch.full((gs.P,), lam, dtype=torch.float64, device=G.device)
        if free_const:  # within (fixed-effects) Grams have a zero constant column: penalise it there
            pen[-1] = 0.0
        A = A + torch.diag(pen)
        chans = np.flatnonzero(levels == lev)
        plain = [c for c in chans if not drop or c not in drop]
        if plain:
            L = torch.linalg.cholesky(A)
            W[plain] = torch.cholesky_solve(R[:, plain] / n, L).T
        for c in chans:
            if drop and c in drop:
                keep = torch.arange(gs.P, device=G.device) != drop[c]
                W[c, keep] = torch.linalg.solve(A[keep][:, keep], R[keep, c] / n)
    return W


def heldout_mse(gs: GramSet, W: torch.Tensor, levels: np.ndarray) -> torch.Tensor:
    """(C,) mean squared error of y - phi W^T on a held-out GramSet, in closed form."""
    out = torch.zeros(gs.C, dtype=torch.float64, device=gs.G.device)
    for lev in np.unique(levels):
        G, R, Y2, n = gs.cumulative(int(lev))
        for c in np.flatnonzero(levels == lev):
            w = W[c]
            out[c] = (Y2[c] - 2 * w @ R[:, c] + w @ G @ w) / n
    return out


class StencilStep(torch.nn.Module):
    """A one-day step dx = W phi(x) [+ b] in standardized units, with the duck-typed surface rollout_rea.unroll calls on a
    neural-lam ForecasterModule: device, state_mean, state_std, on_after_batch_transfer and common_step. Like
    neural-lam, the boundary band takes the truth every step; points where a channel is not real keep the truth too
    (a constant in the pack)."""

    def __init__(self, W, b, shape, k, keep_truth, state_mean, state_std, forcing_mean, forcing_std, forcing: bool = True):
        super().__init__()
        self.register_buffer("forcing_mean", torch.as_tensor(forcing_mean, dtype=torch.float32))
        self.register_buffer("forcing_std", torch.as_tensor(forcing_std, dtype=torch.float32))
        self.shape, self.k, self.use_forcing = tuple(shape), int(k), forcing
        self.register_buffer("W", torch.as_tensor(W, dtype=torch.float32))
        self.register_buffer("b", torch.as_tensor(b, dtype=torch.float32) if b is not None else torch.zeros(1))
        self.register_buffer("keep_truth", torch.as_tensor(keep_truth, dtype=torch.bool))
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.as_tensor(state_std, dtype=torch.float32))

    @property
    def device(self):
        return self.W.device

    def step(self, x: torch.Tensor, forcing: torch.Tensor | None) -> torch.Tensor:
        phi = features(x, forcing if self.use_forcing else None, self.shape, self.k)
        return x + phi @ self.W.T + self.b

    def on_after_batch_transfer(self, batch, idx):
        """Standardize as neural-lam's ForecasterModule does: WeatherDataset returns physical units, and the forcing
        is (..., feature x window), feature-major."""
        init, target, forcing, times = batch
        w = forcing.shape[-1] // self.forcing_mean.numel()
        fm, fs = self.forcing_mean.repeat_interleave(w), self.forcing_std.repeat_interleave(w)
        return ((init - self.state_mean) / self.state_std, (target - self.state_mean) / self.state_std,
                (forcing - fm) / fs, times)

    def common_step(self, batch):
        init, target, forcing, times = batch
        x, out = init[:, 1], []
        for l in range(target.shape[1]):
            x = torch.where(self.keep_truth, target[:, l], self.step(x, forcing[:, l]))
            out.append(x)
        return torch.stack(out, 1), target, None, times


def save_module(path: Path, W, b, shape, k, forcing: bool, info: dict) -> None:
    torch.save({"kind": KIND, "W": torch.as_tensor(W).float().cpu(), "b": None if b is None else torch.as_tensor(b).float().cpu(),
                "shape": tuple(shape), "k": int(k), "forcing": forcing, "info": info}, path)


def is_stencil(path: Path) -> bool:
    if Path(path).suffix != ".pt":
        return False
    return torch.load(path, map_location="cpu", weights_only=False).get("kind") == KIND


def load_module(path: Path, ds, device=None) -> StencilStep:
    """A StencilStep for datastore ds (its boundary mask, level_ocean and standardization)."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return module_for(ds, d["W"], d["b"], d["shape"], d["k"], d["forcing"], device)


def module_for(ds, W, b, shape, k, forcing: bool, device) -> StencilStep:
    """A StencilStep with ds's boundary band, real-point masks and standardization."""
    meta = ds.meta
    names = ds.get_vars_names("state")
    lay = Layout.build(meta.level_ocean.values, names, [float(z) for z in meta.level.values], shape, k)
    keep_truth = np.asarray(ds.boundary_mask.values, bool)[:, None] | ~lay.valid()
    st, fo = ds.get_standardization_dataarray("state"), ds.get_standardization_dataarray("forcing")
    return StencilStep(W, b, shape, k, keep_truth, st.state_mean.values, st.state_std.values, fo.forcing_mean.values,
                       fo.forcing_std.values, forcing).to(device).eval()


def describe(path: Path) -> str:
    d = torch.load(path, map_location="cpu", weights_only=False)
    return json.dumps({"k": d["k"], "forcing": d["forcing"], "W": list(d["W"].shape),
                       "b": None if d["b"] is None else list(d["b"].shape), **d["info"]}, indent=1)


# --- fitting ------------------------------------------------------------------------------------------------------

def _days(times: np.ndarray, start: str, end: str, every: int) -> list[int]:
    t = times.astype("datetime64[D]")
    ok = np.flatnonzero((t >= np.datetime64(start)) & (t <= np.datetime64(end)))
    return [int(i) for i in ok[::every]]


def accumulate(config: Path, out: Path, start: str, end: str, every: int, k: int, target: str = "diff",
               per_cell: bool = False, forcing: bool = True, device: str | None = None, limit: int | None = None) -> GramSet:
    """Grams of one-day pairs whose target day lies in [start, end], every `every` days, from the datastore's train split.
    target 'diff': phi = features(x_t, forcing), y = x_{t+1} - x_t (the stencil step). target 'state': phi = window(x_t)
    and a constant, y = x_t (the spatial conditionals of the realism energy). All in standardized units."""
    from hycom_emulator.evaluate_b00 import load

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds, data, _ = load(config, None, "train", 1)
    meta = ds.meta
    names = ds.get_vars_names("state")
    shape = (ds.grid_shape_state.x, ds.grid_shape_state.y)
    lay = Layout.build(meta.level_ocean.values, names, [float(z) for z in meta.level.values], shape, k)
    rows = rows_by_class(lay, device=device)
    from hycom_emulator.datastore import kept_windows

    # sample i holds rows i, i + 1 (init) and i + 2 (target); drop windows that straddle a source change
    tt = ds.get_dataarray("state", "train").time.values
    times = tt[2:2 + len(data)]
    kept = set(kept_windows(tt, 3, len(data)).tolist())
    idx = [i for i in _days(times, start, end, every) if i in kept][:limit]
    C = len(names)
    nf = data[0][2].shape[-1] if (forcing and target == "diff") else 0
    P = C * k * k + nf + 1
    gs = GramSet(P, C, lay.nclass, per_cell=per_cell).to(device)
    std = module_for(ds, torch.zeros(C, P), None, shape, k, bool(nf), device)
    for j, i in enumerate(idx):
        init, tgt, frc, tm = data[i]
        assert np.datetime64(int(tm[-1]), "ns") == times[i], (i, tm, times[i])
        init, tgt, frc, _ = std.on_after_batch_transfer(tuple(a[None].to(device) for a in (init, tgt, frc, tm)), 0)
        x = init[:, 1]
        if target == "diff":
            phi = features(x, frc[:, -1] if nf else None, shape, k)[0]
            y = (tgt[0, -1] - x[0])
        else:
            phi = features(x, None, shape, k)[0]
            y = x[0]
        gs.add(phi, y, rows)
        if j % 50 == 0:
            print(f"accumulate {target}: {j + 1}/{len(idx)} {str(times[i])[:10]}", flush=True)
    gs.meta = {"config": str(config), "start": start, "end": end, "every": every, "k": k, "target": target,
               "forcing": bool(nf), "days": len(idx), "first": str(times[idx[0]])[:10], "last": str(times[idx[-1]])[:10],
               "names": names, "levels": lay.levels.tolist(), "shape": list(shape)}
    gs.to("cpu").save(out)
    return gs


def fit(train: Path, held: Path, out_dir: Path, lambdas: list[float], device: str | None = None) -> dict:
    """Choose one ridge lambda by the held-out Grams (mean over channels of held-out MSE / held-out mean square of the
    target), then write stencil-shared.pt (constant only) and, when the train Grams carry per-cell sums,
    stencil-bias.pt (CRAM's per-cell bias, the exact fixed-effects solve at the same lambda)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr, ho = GramSet.load(train).to(device), GramSet.load(held).to(device)
    levels = np.array(tr.meta["levels"])
    shape, k, frc = tuple(tr.meta["shape"]), tr.meta["k"], tr.meta["forcing"]
    scale = torch.stack([ho.cumulative(int(levels[c]))[2][c] / ho.cumulative(int(levels[c]))[3] for c in range(tr.C)])
    table = []
    for lam in lambdas:
        W = ridge(tr, levels, lam)
        mse = heldout_mse(ho, W, levels)
        table.append({"lambda": lam, "score": float((mse / scale).mean()), "mse": mse.cpu().tolist()})
        print(f"lambda {lam:g}: held-out MSE / mean square {table[-1]['score']:.4f}", flush=True)
    best = min(table, key=lambda r: r["score"])
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"lambda": best["lambda"], "heldout_score": best["score"], "train": tr.meta, "held": {k_: v for k_, v in ho.meta.items() if k_ != "names"},
            "persistence_score": 1.0}
    W = ridge(tr, levels, best["lambda"])
    save_module(out_dir / "stencil-shared.pt", W, None, shape, k, frc, {**info, "arm": "shared"})
    res = {"lambdas": [{k_: v for k_, v in r.items() if k_ != "mse"} for r in table], "best": best["lambda"],
           "heldout_mse_per_channel": best["mse"], "names": tr.meta["names"]}
    if tr.per_cell:
        rows = _rows_from_meta(tr, device)  # the rows accumulate used: every non-edge cell of each class
        Wb = ridge(tr.within(rows), levels, best["lambda"], free_const=False)
        d = float(tr.days)
        b = (tr.S_y / d - (tr.S_phi / d) @ Wb.T)  # (grid, C): each cell's mean target minus the fitted mean
        valid = torch.as_tensor(_valid_from_meta(tr), device=device)
        b = torch.where(valid, b, torch.zeros_like(b))
        Wb[:, -1] = 0.0  # the per-cell bias carries the constant
        save_module(out_dir / "stencil-bias.pt", Wb, b, shape, k, frc, {**info, "arm": "bias"})
        res["bias_abs_mean"] = float(b.abs().mean())
    (out_dir / "fit.json").write_text(json.dumps(res, indent=1))
    return res


def _layout_from_meta(gs: GramSet) -> Layout:
    import xarray as xr
    import yaml

    cfg = yaml.safe_load(Path(gs.meta["config"]).read_text())
    store = Path(gs.meta["config"]).parent / cfg["datastore"]["config_path"]
    zarr = Path(yaml.safe_load(store.read_text())["zarr"])
    meta = xr.open_zarr(zarr / "meta.zarr", consolidated=True)
    return Layout.build(meta.level_ocean.values, gs.meta["names"], [float(z) for z in meta.level.values], tuple(gs.meta["shape"]), gs.meta["k"])


def _rows_from_meta(gs: GramSet, device) -> list:
    return rows_by_class(_layout_from_meta(gs), device=device)


def _valid_from_meta(gs: GramSet) -> np.ndarray:
    return _layout_from_meta(gs).valid()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("accumulate", help="Grams of one-day pairs from the datastore's train split")
    a.add_argument("config", type=Path)
    a.add_argument("out", type=Path)
    a.add_argument("--start", required=True)
    a.add_argument("--end", required=True)
    a.add_argument("--every", type=int, default=1)
    a.add_argument("--k", type=int, default=5)
    a.add_argument("--target", choices=("diff", "state"), default="diff")
    a.add_argument("--per-cell", action="store_true")
    a.add_argument("--no-forcing", action="store_true")
    a.add_argument("--limit", type=int)
    f = sub.add_parser("fit", help="ridge solve with lambda chosen on held-out Grams; writes stencil-*.pt")
    f.add_argument("train", type=Path)
    f.add_argument("held", type=Path)
    f.add_argument("out_dir", type=Path)
    f.add_argument("--lambdas", default="1e-4,1e-3,1e-2,1e-1,1")
    d = sub.add_parser("describe")
    d.add_argument("path", type=Path)
    args = p.parse_args()
    if args.cmd == "accumulate":
        accumulate(args.config, args.out, args.start, args.end, args.every, args.k, args.target, args.per_cell,
                   not args.no_forcing, limit=args.limit)
    elif args.cmd == "fit":
        print(json.dumps({k_: v for k_, v in fit(args.train, args.held, args.out_dir, [float(x) for x in args.lambdas.split(",")]).items()
                          if k_ != "heldout_mse_per_channel"}, indent=1))
    else:
        print(describe(args.path))


if __name__ == "__main__":
    main()
