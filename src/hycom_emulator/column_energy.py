"""A CRAM-style discrete column energy: sample subsurface T/S profiles given SSH and SST.

Thesis-side probe linked to NeSPReSO (T/S profiles from satellite SSH + SST). The method follows CRAM
(arXiv 2611.17309, "Classical Energies Learn to Reason 1,000x Faster than Looped Models"), mapped so:

- cell: one scalar of an ocean column. N = 45 cells: temperature at the 22 pack levels (cells 0-21),
  salinity at the 22 levels (22-43) and SSH (44). A column of the reanalysis is one "puzzle".
- symbol: each cell's value is discretized into Q quantile bins (default 24), edges per cell from train data.
  A bin maps back to a value through its train median.
- energy: E(x) = sum_i B_i[x_i] + 1/2 sum_{i != j} K_ij[x_i, x_j], a Q x Q table per cell pair with
  K_ij = K_ji^T, stored as one symmetric (N Q) x (N Q) matrix with zero diagonal blocks. With --season, B_i is
  affine in (sin, cos) of day of year.
- training: pseudolikelihood only. For every cell, the conditional softmax over its Q symbols of -E with all
  other cells at their true values, cross-entropy to the true symbol, averaged over cells. No sampling, no
  unrolling. Adam on the smooth part, then a proximal L1 soft-threshold of lr * l1 on K for sparsity. Adam
  steps are at most about lr, so l1 in (0, 1) acts as a threshold on gradient-sign consistency: 1e-5 prunes
  nothing, 0.3 zeroes about 40% of the coupling on the dev subset.
- inference: the observed cells (SSH and T at 0 m) are clamped to their bins, free cells start uniform random,
  and systematic-scan Gibbs at temperature 1 samples the posterior (not its minimum). Several chains, several
  thinned draws per chain after burn-in, give an ensemble of profiles per column.

Baselines: climatology (per-cell train quantiles as an ensemble, ignoring the inputs) and a Gaussian linear
regression of the 44 T/S values on (1, SSH, SST, sin doy, cos doy) with a Gaussian residual (per-level
predictive N(mu, sd^2), closed-form CRPS). Metrics per variable and depth: RMSE of the ensemble or predictive
mean, fair CRPS (ensemble) or Gaussian CRPS, and spread/error ratio. The reason a discrete energy might beat the
Gaussian is 100-500 m temperature, where Loop Current against Gulf common water makes the conditional bimodal.

Data: a reanalysis pack directory (state.npy memmap (days, grid, 91) + meta.zarr) or the dev subset .npz
(keys rows, days, state) with --meta. Columns: all 22 levels ocean, static gulf and ocean, outside boundary_mask;
gap-filled days (meta time_filled) are skipped. Columns are sampled per day.

    python -m hycom_emulator.column_energy fit SOURCE OUT.pt [--meta M] [--dates train] [--per-day 200] ...
    python -m hycom_emulator.column_energy evaluate OUT.pt SOURCE [--meta M] [--dates val] [--out m.json] ...

--dates takes a split name (train 2001-2020, heldout 2021, val 2022-2023, test 2024) or START:END.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr

SPLITS = {
    "train": ("2001-01-01", "2020-12-31"),
    "heldout": ("2021-01-01", "2021-12-31"),
    "val": ("2022-01-01", "2023-12-31"),
    "test": ("2024-01-01", "2024-12-31"),
}
OBSERVED = ("temp_0m", "ssh")


# ---------------------------------------------------------------- data


@dataclass
class Source:
    state: np.ndarray  # (days, grid, features), raw units; memmap for a pack
    days: np.ndarray  # datetime64[D], one per state day
    filled: np.ndarray  # bool per state day
    features: list[str]
    columns: np.ndarray  # grid indices of usable columns
    depths: list[float]

    @property
    def cells(self) -> list[str]:
        return cell_names(self.depths)


def cell_names(depths) -> list[str]:
    tags = [f"{d:g}m" for d in depths]
    return [f"temp_{t}" for t in tags] + [f"salin_{t}" for t in tags] + ["ssh"]


def open_source(path: str | Path, meta: str | Path | None = None) -> Source:
    path = Path(path)
    if path.suffix == ".npz":
        m = xr.open_zarr(meta or path.with_name("meta.zarr"), consolidated=True).load()
        d = np.load(path)
        state, days = d["state"], d["days"].astype("datetime64[D]")
        filled = m.time_filled.values[d["rows"]]
    else:
        m = xr.open_zarr(meta or path / "meta.zarr", consolidated=True).load()
        state = np.load(path / "state.npy", mmap_mode="r")
        days, filled = m.time.values.astype("datetime64[D]"), m.time_filled.values
    if len(days) != state.shape[0]:
        raise ValueError(f"{len(days)} days in meta, {state.shape[0]} in state")
    static = {str(f): m.static.sel(static_feature=f).values > 0.5 for f in ("gulf", "ocean")}
    usable = m.level_ocean.values.astype(bool).all(1) & static["gulf"] & static["ocean"] & (m.boundary_mask.values == 0)
    return Source(state, days, filled.astype(bool), [str(f) for f in m.state_feature.values],
                  np.flatnonzero(usable), [float(z) for z in m.level.values])


def date_range(spec: str) -> tuple[np.datetime64, np.datetime64]:
    a, b = SPLITS[spec] if spec in SPLITS else spec.split(":")
    return np.datetime64(a, "D"), np.datetime64(b, "D")


def season_features(days: np.ndarray) -> np.ndarray:
    doy = (days - days.astype("datetime64[Y]")).astype(int)
    ang = 2 * np.pi * doy / 365.25
    return np.stack([np.sin(ang), np.cos(ang)], -1).astype(np.float32)


def sample_columns(src: Source, dates: str, per_day: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(values (n, 45) float32 in cell order, day (n,) datetime64[D]) for per_day random columns per day."""
    a, b = date_range(dates)
    take = np.flatnonzero((src.days >= a) & (src.days <= b) & ~src.filled)
    if take.size == 0:
        raise ValueError(f"no days in {dates}")
    ch = [src.features.index(c) for c in src.cells]
    rng = np.random.default_rng(seed)
    vals, days = [], []
    for t in take:
        cols = np.sort(rng.choice(src.columns, min(per_day, src.columns.size), replace=False))
        vals.append(np.asarray(src.state[t][cols][:, ch], np.float32))
        days.append(np.full(cols.size, src.days[t]))
    return np.concatenate(vals), np.concatenate(days)


# ---------------------------------------------------------------- discretization


def fit_edges(vals: np.ndarray, q: int) -> np.ndarray:
    """(cells, q - 1) interior quantile edges."""
    return np.quantile(vals, np.linspace(0, 1, q + 1)[1:-1], axis=0).T.astype(np.float32)


def to_bins(vals: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.stack([np.searchsorted(edges[i], vals[:, i], side="right") for i in range(edges.shape[0])], 1)


def bin_medians(vals: np.ndarray, bins: np.ndarray, q: int) -> np.ndarray:
    """(cells, q) train median of each bin; an empty bin (tied edges) takes its nearest non-empty neighbours'."""
    med = np.full((vals.shape[1], q), np.nan, np.float32)
    for i in range(vals.shape[1]):
        order = np.argsort(bins[:, i], kind="stable")
        b, v = bins[order, i], vals[order, i]
        starts = np.searchsorted(b, np.arange(q + 1))
        for k in range(q):
            if starts[k + 1] > starts[k]:
                med[i, k] = np.median(v[starts[k]:starts[k + 1]])
        ok = ~np.isnan(med[i])
        med[i] = np.interp(np.arange(q), np.flatnonzero(ok), med[i, ok])
    return med


# ---------------------------------------------------------------- energy


class ColumnEnergy(torch.nn.Module):
    """E(x) = sum_i B_i[x_i] + 1/2 sum_{i != j} K_ij[x_i, x_j] over N cells with Q symbols each."""

    def __init__(self, n: int, q: int, season: bool = False):
        super().__init__()
        self.n, self.q = n, q
        self.W = torch.nn.Parameter(torch.zeros(n * q, n * q))
        self.B = torch.nn.Parameter(torch.zeros(n, q))
        self.S = torch.nn.Parameter(torch.zeros(2, n, q)) if season else None
        block = torch.arange(n).repeat_interleave(q)
        self.register_buffer("off", (block[:, None] != block[None, :]).float())

    def kmat(self) -> torch.Tensor:
        """Symmetric (N Q, N Q) coupling, zero within a cell: row block i, column block j is K_ij."""
        return 0.5 * (self.W + self.W.T) * self.off

    def bias(self, season: torch.Tensor | None, m: int) -> torch.Tensor:
        """(m, N, Q) per-sample bias."""
        if self.S is None or season is None:
            return self.B.expand(m, -1, -1)
        return self.B + torch.einsum("ms,snq->mnq", season, self.S)

    def onehot(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(x, self.q).float().reshape(x.shape[0], -1)

    def logits(self, x: torch.Tensor, season: torch.Tensor | None = None) -> torch.Tensor:
        """(m, N, Q): every cell's conditional log-probability up to a constant, -E with the others fixed."""
        field = (self.onehot(x) @ self.kmat()).reshape(-1, self.n, self.q)
        return -(self.bias(season, x.shape[0]) + field)

    def energy(self, x: torch.Tensor, season: torch.Tensor | None = None) -> torch.Tensor:
        X = self.onehot(x)
        b = self.bias(season, x.shape[0]).gather(2, x[..., None]).sum((1, 2))
        return b + 0.5 * ((X @ self.kmat()) * X).sum(1)

    def pl_loss(self, x: torch.Tensor, season: torch.Tensor | None = None) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(self.logits(x, season).reshape(-1, self.q), x.reshape(-1))


def train_energy(bins: np.ndarray, q: int, season: np.ndarray | None = None, steps: int = 1000, batch: int = 2048,
                 lr: float = 1e-2, l1: float = 0.3, seed: int = 0, val_frac: float = 0.05, log_every: int = 100):
    """Pseudolikelihood fit with Adam and a proximal L1 step on the coupling. Returns (model, log)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.as_tensor(bins, dtype=torch.long)
    s = None if season is None else torch.as_tensor(season, dtype=torch.float32)
    perm = torch.randperm(x.shape[0], generator=g)
    nval = int(val_frac * x.shape[0])
    vi, ti = perm[:nval], perm[nval:]
    model = ColumnEnergy(x.shape[1], q, season is not None)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    for step in range(1, steps + 1):
        idx = ti[torch.randint(ti.numel(), (min(batch, ti.numel()),), generator=g)]
        loss = model.pl_loss(x[idx], None if s is None else s[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():  # prox of lr * l1 * |W|_1
            model.W.copy_(torch.sign(model.W) * (model.W.abs() - lr * l1).clamp_min(0))
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                val = model.pl_loss(x[vi], None if s is None else s[vi]).item() if nval else float("nan")
            log.append({"step": step, "train_pl": loss.item(), "val_pl": val})
    return model, log


@torch.no_grad()
def gibbs(model: ColumnEnergy, x0: torch.Tensor, clamp: torch.Tensor, season: torch.Tensor | None = None,
          chains: int = 8, burn: int = 100, keep: int = 4, thin: int = 10, seed: int = 0) -> torch.Tensor:
    """(m, chains * keep, N) posterior draws at temperature 1. Cells where clamp (N,) is True keep x0's symbols;
    free cells start uniform random. Systematic scan; after burn sweeps, one draw every thin sweeps per chain."""
    g = torch.Generator().manual_seed(seed)
    n, q = model.n, model.q
    x = x0.repeat_interleave(chains, 0).clone()
    free = torch.nonzero(~clamp).flatten().tolist()
    x[:, free] = torch.randint(q, (x.shape[0], len(free)), generator=g)
    X = model.onehot(x)
    kcols = model.kmat().T.reshape(n, q, n * q).contiguous()
    bias = model.bias(None if season is None else season.repeat_interleave(chains, 0), x.shape[0])
    draws = []
    for sweep in range(burn + keep * thin):
        for i in free:
            logits = -(bias[:, i] + X @ kcols[i].T)
            u = torch.rand(logits.shape, generator=g).clamp_min(1e-12)
            v = torch.argmax(logits - torch.log(-torch.log(u)), 1)
            x[:, i] = v
            X[:, i * q:(i + 1) * q] = torch.nn.functional.one_hot(v, q).float()
        if sweep >= burn and (sweep - burn + 1) % thin == 0:
            draws.append(x.clone())
    return torch.stack(draws, 1).reshape(x0.shape[0], chains * keep, n)


# ---------------------------------------------------------------- baselines and scores


def gaussian_features(vals: np.ndarray, season: np.ndarray, cells: list[str]) -> np.ndarray:
    obs = vals[:, [cells.index("ssh"), cells.index("temp_0m")]]
    return np.concatenate([np.ones((len(vals), 1)), obs, season], 1).astype(np.float64)


def fit_gaussian(vals: np.ndarray, season: np.ndarray, cells: list[str], ntarget: int):
    F, Y = gaussian_features(vals, season, cells), vals[:, :ntarget].astype(np.float64)
    beta = np.linalg.lstsq(F, Y, rcond=None)[0]
    r = Y - F @ beta
    cov = r.T @ r / (len(Y) - F.shape[1])
    return beta, cov


def fair_crps(ens: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(n, C) fair CRPS of an (n, M, C) ensemble: E|X - y| - sum_{i,j}|X_i - X_j| / (2 M (M - 1))."""
    m = ens.shape[1]
    xs = np.sort(ens, 1)
    w = (2 * np.arange(1, m + 1) - m - 1)[None, :, None]
    pair = 2 * (w * xs).sum(1)  # sum over ordered pairs of |X_i - X_j|
    return np.abs(ens - y[:, None]).mean(1) - pair / (2 * m * (m - 1))


def gaussian_crps(mu, sd, y):
    """CRPS of N(mu, sd^2) at y: sd [z (2 Phi(z) - 1) + 2 phi(z) - 1 / sqrt(pi)]."""
    z = (y - mu) / sd
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + torch.erf(torch.as_tensor(z / math.sqrt(2), dtype=torch.float64)).numpy())
    return sd * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def ens_scores(ens: np.ndarray, y: np.ndarray) -> dict:
    m = ens.shape[1]
    rmse = np.sqrt(((ens.mean(1) - y) ** 2).mean(0))
    spread = np.sqrt(ens.var(1, ddof=1).mean(0) * (m + 1) / m)
    return {"rmse": rmse, "crps": fair_crps(ens, y).mean(0), "ratio": spread / rmse}


def gauss_scores(mu: np.ndarray, sd: np.ndarray, y: np.ndarray) -> dict:
    rmse = np.sqrt(((mu - y) ** 2).mean(0))
    return {"rmse": rmse, "crps": gaussian_crps(mu, sd, y).mean(0), "ratio": np.sqrt((sd ** 2).mean(0)) / rmse}


# ---------------------------------------------------------------- CLI


def fit(args) -> None:
    src = open_source(args.source, args.meta)
    t0 = time.time()
    vals, days = sample_columns(src, args.dates, args.per_day, args.seed)
    season = season_features(days)
    t_load = time.time() - t0
    edges = fit_edges(vals, args.q)
    bins = to_bins(vals, edges)
    med = bin_medians(vals, bins, args.q)
    t0 = time.time()
    model, log = train_energy(bins, args.q, season if args.season else None, args.steps, args.batch, args.lr,
                              args.l1, args.seed)
    t_train = time.time() - t0
    ntarget = 2 * len(src.depths)
    beta, cov = fit_gaussian(vals, season, src.cells, ntarget)
    clim = np.quantile(vals[:, :ntarget], (np.arange(args.clim_members) + 0.5) / args.clim_members, axis=0)
    with torch.no_grad():
        k = model.kmat()
    nz = float((k != 0).float().sum() / model.off.sum())
    info = {"columns": int(len(vals)), "days": int(np.unique(days).size), "load_s": t_load, "train_s": t_train,
            "coupling_nonzero": nz, "log": log}
    torch.save({"state_dict": model.state_dict(), "n": model.n, "q": args.q, "season": args.season,
                "cells": src.cells, "edges": edges, "medians": med, "gauss_beta": beta, "gauss_cov": cov,
                "clim": clim.astype(np.float32), "info": info, "args": vars(args)}, args.out)
    print(json.dumps({k: v for k, v in info.items() if k != "log"} | {"last": log[-1]}, indent=1))


def evaluate(args) -> None:
    ck = torch.load(args.ckpt, weights_only=False)
    model = ColumnEnergy(ck["n"], ck["q"], ck["season"])
    model.load_state_dict(ck["state_dict"])
    cells, med = ck["cells"], ck["medians"]
    src = open_source(args.source, args.meta)
    if src.cells != cells:
        raise ValueError("checkpoint cells differ from the source's")
    vals, days = sample_columns(src, args.dates, args.per_day, args.seed + 1)
    season = season_features(days)
    ntarget = ck["gauss_beta"].shape[1]
    y = vals[:, :ntarget]
    bins = to_bins(vals, ck["edges"])
    clamp = torch.zeros(len(cells), dtype=torch.bool)
    clamp[[cells.index(c) for c in OBSERVED]] = True
    t0 = time.time()
    ens = []
    for a in range(0, len(vals), args.batch):
        xb = torch.as_tensor(bins[a:a + args.batch], dtype=torch.long)
        sb = torch.as_tensor(season[a:a + args.batch]) if ck["season"] else None
        d = gibbs(model, xb, clamp, sb, args.chains, args.burn, args.keep, args.thin, args.seed + a).numpy()
        ens.append(np.take_along_axis(med[None, None, :, :], d[..., None], 3)[..., 0][:, :, :ntarget])
    ens = np.concatenate(ens)
    t_gibbs = time.time() - t0
    mu = gaussian_features(vals, season, cells) @ ck["gauss_beta"]
    sd = np.sqrt(np.diag(ck["gauss_cov"]))[None].repeat(len(y), 0)
    clim = np.broadcast_to(ck["clim"][None], (len(y),) + ck["clim"].shape)
    quant = np.take_along_axis(med, bins.T, 1).T[:, :ntarget]
    scores = {"energy": ens_scores(ens, y), "gauss": gauss_scores(mu, sd, y), "clim": ens_scores(clim, y),
              "bin_floor": {"rmse": np.sqrt(((quant - y) ** 2).mean(0))}}
    table = {m: {k: dict(zip(cells[:ntarget], np.asarray(v, float).round(5).tolist())) for k, v in s.items()}
             for m, s in scores.items()}
    t_band = [c for c in cells[:ntarget] if c.startswith("temp_") and 100 <= float(c[5:-1]) <= 500]
    band = {m: {k: float(np.mean([table[m][k][c] for c in t_band])) for k in table[m]} for m in table}
    out = {"columns": int(len(y)), "days": int(np.unique(days).size), "members": int(ens.shape[1]),
           "gibbs": {"chains": args.chains, "burn": args.burn, "keep": args.keep, "thin": args.thin,
                     "sweeps": args.burn + args.keep * args.thin, "seconds": t_gibbs},
           "temp_100_500m": band, "per_cell": table}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    print(summary(out))


def summary(out: dict, depths=(0, 50, 100, 200, 300, 400, 500, 1000)) -> str:
    g = out["gibbs"]
    lines = [f"{out['columns']} columns, {out['days']} days, {out['members']} members; Gibbs {g['chains']} chains, "
             f"burn {g['burn']}, keep {g['keep']} every {g['thin']} ({g['sweeps']} sweeps, {g['seconds']:.0f} s)",
             f"{'cell':>12} | {'rmse E/G/C/floor':>32} | {'crps E/G/C':>24} | {'spread/err E/G/C':>18}"]
    pc = out["per_cell"]
    rows = [f"{v}_{d}m" for v in ("temp", "salin") for d in depths]
    for c in rows + ["temp_100_500m"]:
        src = {m: ({k: v[c] for k, v in pc[m].items()} if c in pc["gauss"]["rmse"] else out["temp_100_500m"][m])
               for m in pc}
        r = " ".join(f"{src[m]['rmse']:.3f}" for m in ("energy", "gauss", "clim", "bin_floor"))
        cr = " ".join(f"{src[m]['crps']:.3f}" for m in ("energy", "gauss", "clim"))
        sp = " ".join(f"{src[m]['ratio']:.2f}" for m in ("energy", "gauss", "clim"))
        lines.append(f"{c:>12} | {r:>32} | {cr:>24} | {sp:>18}")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="column_energy", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("source")
    f.add_argument("out")
    f.add_argument("--meta")
    f.add_argument("--dates", default="train")
    f.add_argument("--per-day", type=int, default=200)
    f.add_argument("--q", type=int, default=24)
    f.add_argument("--season", action="store_true")
    f.add_argument("--steps", type=int, default=1000)
    f.add_argument("--batch", type=int, default=2048)
    f.add_argument("--lr", type=float, default=1e-2)
    f.add_argument("--l1", type=float, default=0.3)
    f.add_argument("--clim-members", type=int, default=32)
    f.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("evaluate")
    e.add_argument("ckpt")
    e.add_argument("source")
    e.add_argument("--meta")
    e.add_argument("--dates", default="val")
    e.add_argument("--per-day", type=int, default=100)
    e.add_argument("--chains", type=int, default=8)
    e.add_argument("--burn", type=int, default=100)
    e.add_argument("--keep", type=int, default=4)
    e.add_argument("--thin", type=int, default=10)
    e.add_argument("--batch", type=int, default=1024)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out")
    args = p.parse_args(argv)
    {"fit": fit, "evaluate": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
