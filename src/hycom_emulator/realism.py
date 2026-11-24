"""Realism energy: a typical-set test of a single ocean state, the Gaussian pseudolikelihood of CRAM (arXiv 2611.17309).

Each state channel c at cell i is predicted from every channel on the k x k window around i with the cell's own column
removed (stencil.window minus the centre of every channel), by one ridge regression per level class with weights shared
by all cells. The residual r_c(i) = x_c(i) - w_c . phi(i), in standardized units, divided by its held-out standard
deviation sigma_c, gives the energy of a state per field group (temp, salin, u, v, ssh, ubaro/vbaro as 'baro'):

    E_f(x) = mean over the group's channels c and the region's real points i of (r_c(i) / sigma_c)^2

On truth E is near 1 by construction. It is not minimised: a squared-error forecast that has lost unpredictable small
scales is locally smoother than any real state, so its E falls BELOW the truth's range (CRAM's Maze lesson: the empty
board has lower energy than the solution). A state with injected noise or a growing instability sits ABOVE it. The test
is two-sided: a state is realistic for a field when log E_f lies inside [q_lo, q_hi] of the truth's log E_f on held-out
days.

    python -m hycom_emulator.realism fit <train_grams.pt> <held_grams.pt> <out.pt> [--lambda L]
    python -m hycom_emulator.realism band <nlam.yaml> <energy.pt> <out.json> --start D --end D [--every N]
    python -m hycom_emulator.realism score <nlam.yaml> <energy.pt> <band.json> <out_dir> --model NAME=<ckpt> ... --starts ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hycom_emulator.forcing_mode import Forcing
from hycom_emulator.stencil import GramSet, Layout, centre, features, heldout_mse, ridge

GROUPS = ("temp", "salin", "u", "v", "ssh", "baro")
KIND = "realism"


def group_of(name: str) -> str:
    if name in ("ubaro", "vbaro"):
        return "baro"
    return name.rpartition("_")[0] if "_" in name else name


def centre_columns(C: int, k: int) -> list[int]:
    return [centre(c, k) for c in range(C)]


def fit_energy(train: GramSet, held: GramSet, lam: float) -> dict:
    """Weights with every channel's centre feature excluded, and sigma_c from the held-out Grams."""
    levels = np.array(train.meta["levels"])
    k, C = train.meta["k"], train.C
    drop = centre_columns(C, k)
    keep = torch.ones(train.P, dtype=torch.bool)
    keep[drop] = False
    sub = lambda gs: GramSet(int(keep.sum()), C, gs.L, gs.G[:, keep][:, :, keep], gs.R[:, keep], gs.Y2, gs.n, gs.days)  # noqa: E731
    W = ridge(sub(train), levels, lam)
    mse = heldout_mse(sub(held), W, levels)
    Wfull = torch.zeros(C, train.P, dtype=torch.float64)
    Wfull[:, keep] = W
    return {"kind": KIND, "W": Wfull.float(), "sigma": mse.sqrt().float(), "k": k, "lambda": lam,
            "names": train.meta["names"], "levels": train.meta["levels"], "shape": train.meta["shape"],
            "train": {a: b for a, b in train.meta.items() if a != "names"}, "held": {a: b for a, b in held.meta.items() if a != "names"}}


class Energy:
    """E_f of standardized states (B, grid, C) over a region of cells."""

    def __init__(self, d: dict, region: np.ndarray, level_ocean: np.ndarray, depths: list[float], device):
        self.k, self.shape = d["k"], tuple(d["shape"])
        self.W = d["W"].to(device)
        self.sigma = d["sigma"].to(device)
        names = d["names"]
        lay = Layout.build(level_ocean, names, depths, self.shape, self.k)
        mask = torch.as_tensor(lay.valid() & region[:, None] & ~lay.edge[:, None], device=device)
        groups = [group_of(n) for n in names]
        self.groups = {g: torch.as_tensor([c for c, h in enumerate(groups) if h == g], device=device) for g in GROUPS}
        self.mask = mask.float()

    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """{group: (B,)} energies of standardized states x (B, grid, C)."""
        r = (x - features(x, None, self.shape, self.k) @ self.W.T) / self.sigma
        e = r * r * self.mask
        return {g: e[..., c].sum((1, 2)) / self.mask[:, c].sum() for g, c in self.groups.items()}


def blur(x: torch.Tensor, shape, ocean: torch.Tensor, sigma_cells: float) -> torch.Tensor:
    """Gaussian blur of (B, grid, C) states inside the ocean (normalized convolution), land left as is."""
    import torch.nn.functional as F

    from hycom_emulator.stencil import columns, images

    r = int(np.ceil(3 * sigma_cells))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    g = torch.exp(-0.5 * (t / sigma_cells) ** 2)
    g = g / g.sum()
    img = images(x, shape)
    m = images(ocean[None, :, None].to(x.dtype).expand(1, -1, 1), shape)
    C = img.shape[1]

    def conv(a, ch):
        a = F.conv2d(a, g.view(1, 1, 1, -1).expand(ch, 1, 1, -1), padding=(0, r), groups=ch)
        return F.conv2d(a, g.view(1, 1, -1, 1).expand(ch, 1, -1, 1), padding=(r, 0), groups=ch)

    num, den = conv(img * m, C), conv(m.expand(img.shape[0], 1, -1, -1).contiguous(), 1)
    out = torch.where(m > 0, num / den.clamp_min(1e-6), img)
    return columns(out)


def _standardizer(ds, device):
    st = ds.get_standardization_dataarray("state")
    mean = torch.as_tensor(st.state_mean.values, dtype=torch.float32, device=device)
    std = torch.as_tensor(st.state_std.values, dtype=torch.float32, device=device)
    return lambda x: (x.float() - mean) / std  # noqa: E731


def _region(ds) -> np.ndarray:
    static = ds.meta.static
    st = lambda f: static.sel(static_feature=f).values.astype(bool)  # noqa: E731
    return st("gulf") & st("ocean") & ~ds.meta.boundary_mask.values.astype(bool)


def band(config: Path, energy: Path, out: Path, start: str, end: str, every: int, q: tuple[float, float]) -> dict:
    """The truth's log E per group on held-out days, and its [q_lo, q_hi] quantiles."""
    from hycom_emulator.evaluate_b00 import load

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, data, _ = load(config, None, "train", 1)
    d = torch.load(energy, map_location="cpu", weights_only=False)
    E = Energy(d, _region(ds), ds.meta.level_ocean.values, [float(z) for z in ds.meta.level.values], device)
    z = _standardizer(ds, device)
    tt = ds.get_dataarray("state", "train").time.values
    t = tt.astype("datetime64[D]")
    rows = np.flatnonzero((t >= np.datetime64(start)) & (t <= np.datetime64(end)))[::every]
    vals = {g: [] for g in GROUPS}
    for r in rows:
        x = z(torch.as_tensor(np.asarray(ds.meta.state[int(r)].values), device=device)[None])
        for g, v in E(x).items():
            vals[g].append(float(v[0]))
    res = {"start": start, "end": end, "every": every, "days": int(rows.size), "q": list(q),
           "log_lo": {g: float(np.quantile(np.log(v), q[0])) for g, v in vals.items()},
           "log_hi": {g: float(np.quantile(np.log(v), q[1])) for g, v in vals.items()},
           "median": {g: float(np.median(v)) for g, v in vals.items()}, "values": vals}
    out.write_text(json.dumps(res, indent=1))
    return res


def score(config: Path, energy: Path, band_json: Path, out: Path, models: dict[str, Path], starts: list[str], horizon: int,
          batch: int, blurs=(1.0, 2.0), noise=0.1) -> None:
    """Per model, start, lead and group: E of the forecast; per start and lead also of the truth, the truth blurred by
    each of `blurs` cells and the truth plus white noise of `noise` standardized units. Writes energy.tsv with columns
    source start lead group energy log_energy inside below above."""
    import csv

    from hycom_emulator import rollout_rea as rr

    bnd = json.loads(band_json.read_text())
    lo, hi = bnd["log_lo"], bnd["log_hi"]
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    gen = torch.Generator().manual_seed(0)
    for i, (name, ckpt) in enumerate(models.items()):
        m = rr.load_model(config, ckpt, forcing=Forcing.ORACLE)
        dev = m.module.device
        d = torch.load(energy, map_location="cpu", weights_only=False)
        E = Energy(d, _region(m.datastore), m.meta.level_ocean.values, [float(z) for z in m.meta.level.values], dev)
        z = _standardizer(m.datastore, dev)
        ocean = torch.as_tensor(m.meta.static.sel(static_feature="ocean").values.astype(bool), device=dev)
        todo = sorted(rr.start_set(m.times.astype("datetime64[D]"), horizon, starts), key=lambda s: s.date)
        for b0 in range(0, len(todo), batch):
            for lead, pos, step in rr.unroll(m.module, m.window, m.times, todo[b0:b0 + batch], horizon):
                srcs = {name: z(step.pred)}
                if i == 0:
                    t = z(step.truth)
                    srcs["truth"] = t
                    for s in blurs:
                        srcs[f"blur{s:g}"] = blur(t, m.shape, ocean, s)
                    srcs[f"noise{noise:g}"] = t + noise * torch.randn(t.shape, generator=gen).to(dev)
                for src, x in srcs.items():
                    for g, v in E(x).items():
                        for p, e in zip(pos, v.tolist()):
                            le = float(np.log(e))
                            rows.append([src, str(todo[b0 + p].date), lead, g, f"{e:.6f}", f"{le:.6f}",
                                         int(lo[g] <= le <= hi[g]), int(le < lo[g]), int(le > hi[g])])
            rr.release_pack_pages(m.datastore)
            print(f"{name}: starts {b0 + 1}..{min(b0 + batch, len(todo))} of {len(todo)}", flush=True)
    with (out / "energy.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow("source start lead group energy log_energy inside below above".split())
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("train", type=Path)
    f.add_argument("held", type=Path)
    f.add_argument("out", type=Path)
    f.add_argument("--lambda", dest="lam", type=float, default=1e-3)
    b = sub.add_parser("band")
    b.add_argument("config", type=Path)
    b.add_argument("energy", type=Path)
    b.add_argument("out", type=Path)
    b.add_argument("--start", required=True)
    b.add_argument("--end", required=True)
    b.add_argument("--every", type=int, default=1)
    b.add_argument("--q", default="0.01,0.99")
    s = sub.add_parser("score")
    s.add_argument("config", type=Path)
    s.add_argument("energy", type=Path)
    s.add_argument("band", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--model", action="append", required=True, help="NAME=checkpoint (.ckpt or stencil .pt)")
    s.add_argument("--starts", required=True)
    s.add_argument("--horizon", type=int, default=21)
    s.add_argument("--batch", type=int, default=4)
    a = p.parse_args()
    if a.cmd == "fit":
        d = fit_energy(GramSet.load(a.train), GramSet.load(a.held), a.lam)
        torch.save(d, a.out)
        print(json.dumps({n: round(float(s_), 4) for n, s_ in zip(d["names"], d["sigma"])}, indent=0))
    elif a.cmd == "band":
        r = band(a.config, a.energy, a.out, a.start, a.end, a.every, tuple(float(x) for x in a.q.split(",")))
        print(json.dumps({k: r[k] for k in ("days", "median", "log_lo", "log_hi")}, indent=1))
    else:
        models = dict(m.split("=", 1) for m in a.model)
        score(a.config, a.energy, a.band, a.out, {k: Path(v) for k, v in models.items()}, a.starts.split(","), a.horizon, a.batch)


if __name__ == "__main__":
    main()
