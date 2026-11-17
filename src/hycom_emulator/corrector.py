"""A stochastic corrector of free runs: CRAM's split of a local trained energy and a search at inference, Gaussian case.

For a deterministic free run x_hat (any checkpoint rollout_rea loads) and lead bin b:

1. Mean: y = x_hat + A_b phi(x_hat) + r, with phi the k x k window of x_hat and a constant (stencil.features), A_b one
   ridge regression shared by all cells. Fitted on (forecast, truth) pairs of free runs from train years: a local,
   teacher-forced objective; no rollout is differentiated. emu-rea-blur-budget-r3 found the band slope of truth on
   forecast near 0.72 at 50-100 km on day 7, so A_b is expected to shrink the unpredictable bands.
2. Residual: r(i) given the window of r with the cell's own column removed is Gaussian, mean W_b phi'(r)(i) and
   column covariance Sigma_b (91 x 91), both from train residuals (realism-style conditionals).
3. Members: block Gibbs on r with 9 colours (cells three apart in x and y share no 5 x 5 window), r clamped to 0 on the
   boundary band and below the bottom (the givens, as CRAM clamps a puzzle's clues), starting from 0, `sweeps` sweeps at
   temperature T_b. The chain is linear in its noise, so the spread scales as sqrt(T_b) exactly; T_b is set on
   held-out train-year runs so that spread over error of the ensemble mean is 1.

    python -m hycom_emulator.corrector collect <nlam.yaml> <ckpt> <out_dir> --stage mean|resid --starts ... [--mean mean.pt]
    python -m hycom_emulator.corrector fit-mean <train_dir> <held_dir> <mean.pt>
    python -m hycom_emulator.corrector fit-noise <train_dir> <held_dir> <mean.pt> <noise.pt>
    python -m hycom_emulator.corrector calibrate <nlam.yaml> <ckpt> <mean.pt> <noise.pt> --starts ... (writes T_b into noise.pt)
    python -m hycom_emulator.corrector evaluate <nlam.yaml> <ckpt> <mean.pt> <noise.pt> <out_dir> --starts ... [--energy e.pt --band b.json]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hycom_emulator.stencil import GramSet, Layout, centre, columns, features, heldout_mse, images, ridge, rows_by_class

LEAD_BINS = ((1, 1), (2, 3), (4, 7), (8, 14), (15, 21))
FIELDS = ("temp", "salin", "u", "v", "ssh")


def lead_bin(lead: int) -> int:
    return next(b for b, (lo, hi) in enumerate(LEAD_BINS) if lo <= lead <= hi)


def as_conv(W: torch.Tensor, C: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(C_out, C k k + 1) feature weights -> conv2d weight (C_out, C, k, k) and bias: unfold's order is cross-correlation."""
    return W[:, :C * k * k].reshape(W.shape[0], C, k, k).contiguous(), W[:, -1].contiguous()


def apply(W: torch.Tensor, x: torch.Tensor, shape, k: int) -> torch.Tensor:
    """phi(x) W^T for (B, grid, C) x by convolution (no unfold buffer)."""
    w, b = as_conv(W.to(x.dtype), x.shape[-1], k)
    return columns(F.conv2d(images(x, shape), w, b, padding=k // 2))


class Standardizer:
    def __init__(self, ds, device):
        st = ds.get_standardization_dataarray("state")
        self.mean = torch.as_tensor(st.state_mean.values, dtype=torch.float32, device=device)
        self.std = torch.as_tensor(st.state_std.values, dtype=torch.float32, device=device)

    def z(self, x):
        return (x.float() - self.mean) / self.std

    def raw(self, x):
        return x * self.std + self.mean


def _layout(m, k: int) -> Layout:
    return Layout.build(m.meta.level_ocean.values, m.geo.names, [float(z) for z in m.meta.level.values], m.shape, k)


def _starts(m, starts, horizon):
    from hycom_emulator import rollout_rea as rr

    todo = sorted(rr.start_set(m.times.astype("datetime64[D]"), horizon, starts), key=lambda s: s.date)
    return [s for s in todo if s.horizon >= horizon]


def collect(config: Path, ckpt: Path, out: Path, stage: str, starts: list[str], horizon: int, k: int, batch: int,
            mean: Path | None = None) -> None:
    """Grams per lead bin. stage 'mean': phi(x_hat) against y - x_hat. stage 'resid': with mean.pt applied, the window of
    r = y - x_hat - A phi(x_hat) against r itself (realism-style, for the conditionals and Sigma)."""
    from hycom_emulator import rollout_rea as rr

    m = rr.load_model(config, ckpt)
    dev = m.module.device
    s = Standardizer(m.datastore, dev)
    lay = _layout(m, k)
    rows = rows_by_class(lay, device=dev)
    C = len(m.geo.names)
    P = C * k * k + 1
    A = torch.load(mean, map_location=dev, weights_only=False)["A"] if stage == "resid" else None
    grams = [GramSet(P, C, lay.nclass).to(dev) for _ in LEAD_BINS]
    todo = _starts(m, starts, horizon)
    for b0 in range(0, len(todo), batch):
        for lead, pos, step in rr.unroll(m.module, m.window, m.times, todo[b0:b0 + batch], horizon):
            xh, y = s.z(step.pred), s.z(step.truth)
            jb = lead_bin(lead)
            if stage == "mean":
                phi, t = features(xh, None, m.shape, k), y - xh
            else:
                r = y - xh - apply(A[jb], xh, m.shape, k)
                phi, t = features(r, None, m.shape, k), r
            for i in range(phi.shape[0]):
                grams[jb].add(phi[i], t[i], rows)
        rr.release_pack_pages(m.datastore)
        print(f"collect {stage}: starts {b0 + 1}..{min(b0 + batch, len(todo))} of {len(todo)}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"config": str(config), "ckpt": str(ckpt), "stage": stage, "k": k, "starts": [str(t.date) for t in todo],
            "horizon": horizon, "names": m.geo.names, "levels": lay.levels.tolist(), "shape": list(m.shape)}
    for jb, g in enumerate(grams):
        g.meta = {**meta, "bin": list(LEAD_BINS[jb])}
        g.to("cpu").save(out / f"grams_b{jb}.pt")


def _load_bins(d: Path) -> list[GramSet]:
    return [GramSet.load(d / f"grams_b{jb}.pt") for jb in range(len(LEAD_BINS))]


def fit_mean(train: Path, held: Path, out: Path, lambdas: list[float]) -> dict:
    """A_b per lead bin, lambda per bin from the held-out Grams; also the held-out MSE of no correction (A = 0)."""
    tr, ho = _load_bins(train), _load_bins(held)
    levels = np.array(tr[0].meta["levels"])
    A, info = [], []
    for jb, (t, h) in enumerate(zip(tr, ho)):
        zero = heldout_mse(h, torch.zeros(t.C, t.P, dtype=torch.float64), levels)
        best = None
        for lam in lambdas:
            W = ridge(t, levels, lam)
            score = float((heldout_mse(h, W, levels) / zero).mean())
            if best is None or score < best[0]:
                best = (score, lam, W)
        A.append(best[2].float())
        info.append({"bin": list(LEAD_BINS[jb]), "lambda": best[1], "heldout_mse_over_uncorrected": best[0]})
        print(json.dumps(info[-1]), flush=True)
    torch.save({"A": torch.stack(A), "k": tr[0].meta["k"], "info": info, "train": tr[0].meta["starts"][:3]}, out)
    return {"bins": info}


def fit_noise(train: Path, held: Path, mean: Path, out: Path, lam: float) -> dict:
    """Per lead bin: conditionals W_b of r(i) on its window without the centre column, and the column covariance
    Sigma_b of their residual from the held-out Grams, Sigma[c, c'] on the classes where both channels are real."""
    from hycom_emulator.realism import centre_columns

    tr, ho = _load_bins(train), _load_bins(held)
    levels = np.array(tr[0].meta["levels"])
    k, C, P = tr[0].meta["k"], tr[0].C, tr[0].P
    keep = torch.ones(P, dtype=torch.bool)
    keep[centre_columns(C, k)] = False
    Ws, Ls, info = [], [], []
    for jb, (t, h) in enumerate(zip(tr, ho)):
        sub = GramSet(int(keep.sum()), C, t.L, t.G[:, keep][:, :, keep], t.R[:, keep], t.Y2, t.n, t.days)
        Wk = ridge(sub, levels, lam)
        W = torch.zeros(C, P, dtype=torch.float64)
        W[:, keep] = Wk
        Sel = torch.zeros(C, P, dtype=torch.float64)
        Sel[torch.arange(C), torch.tensor(centre_columns(C, k))] = 1.0
        D = Sel - W  # e = D phi
        S = torch.zeros(C, C, dtype=torch.float64)
        for la in np.unique(levels):
            for lb in np.unique(levels):
                G, _, _, n = h.cumulative(int(max(la, lb)))
                ia, ib = np.flatnonzero(levels == la), np.flatnonzero(levels == lb)
                S[np.ix_(ia, ib)] = D[ia] @ G @ D[ib].T / n
        S = 0.5 * (S + S.T)
        ev, V = torch.linalg.eigh(S)
        L = V * ev.clamp_min(1e-10).sqrt()  # Sigma = L L^T, robust to a near-singular Sigma
        Ws.append(W.float())
        Ls.append(L.float())
        info.append({"bin": list(LEAD_BINS[jb]), "sigma_diag_median": float(S.diagonal().median()), "min_eig": float(ev.min())})
        print(json.dumps(info[-1]), flush=True)
    torch.save({"W": torch.stack(Ws), "L": torch.stack(Ls), "T": torch.ones(len(LEAD_BINS)), "k": k, "lambda": lam,
                "info": info, "mean": str(mean)}, out)
    return {"bins": info}


class Sampler:
    """Block Gibbs on residual fields r (M, grid, C) of one lead bin, 9 colours, clamped to 0 where not free."""

    def __init__(self, noise: dict, jb: int, shape, free: torch.Tensor, valid: torch.Tensor, device):
        self.shape, self.k = shape, noise["k"]
        nx, ny = shape
        ix, iy = np.divmod(np.arange(nx * ny), ny)
        colour = torch.as_tensor((ix % 3) * 3 + (iy % 3), device=device)
        self.cells = [torch.nonzero(free & (colour == q)).squeeze(1) for q in range(9)]
        self.W, self.L = noise["W"][jb].to(device), noise["L"][jb].to(device)
        self.T = float(noise["T"][jb])
        self.valid = valid.float()

    def sample(self, M: int, sweeps: int, gen: torch.Generator, T: float | None = None) -> torch.Tensor:
        T = self.T if T is None else T
        C = self.W.shape[0]
        r = torch.zeros(M, self.valid.shape[0], C, device=self.W.device)
        sq = float(np.sqrt(T))
        for _ in range(sweeps):
            for q in self.cells:
                mu = apply(self.W, r, self.shape, self.k)[:, q]
                eps = torch.randn(M, q.numel(), self.L.shape[1], generator=gen, device="cpu").to(r.device)
                r[:, q] = (mu + sq * eps @ self.L.T) * self.valid[q]
        return r


def _weights(m, f):
    cols, w = m.geo.groups[f]
    return cols, w  # w: (grid, k) float64


def ens_stats(members: torch.Tensor, truth: torch.Tensor, m) -> dict:
    """members (M, grid, C) and truth (grid, C) raw. Per field: ensemble-mean RMSE, spread (sqrt of the weighted mean
    member variance, M/(M-1)), and fair CRPS, all with rollout_rea's column weights."""
    M = members.shape[0]
    out = {}
    for f in FIELDS:
        cols, w = _weights(m, f)
        x, y = members[..., cols].double(), truth[..., cols].double()
        ws = w.sum()
        mean = x.mean(0)
        out[f"rmse_{f}"] = float(((w * (mean - y) ** 2).sum() / ws).sqrt())
        out[f"spread_{f}"] = float(((w * x.var(0, unbiased=True)).sum() / ws).sqrt())
        xs, _ = torch.sort(x, 0)
        rank = torch.arange(1, M + 1, device=x.device, dtype=x.dtype).view(-1, *([1] * (x.dim() - 1)))
        pair = (2 * (2 * rank - M - 1) * xs).sum(0) / (M * (M - 1))  # mean over m != m' of |x_m - x_m'|
        crps = (x - y).abs().mean(0) - 0.5 * pair
        out[f"crps_{f}"] = float((w * crps).sum() / ws)
    return out


def point_stats(x: torch.Tensor, truth: torch.Tensor, m) -> dict:
    out = {}
    for f in FIELDS:
        cols, w = _weights(m, f)
        e = (x[..., cols] - truth[..., cols]).double()
        out[f"rmse_{f}"] = float(((w * e * e).sum() / w.sum()).sqrt())
        out[f"crps_{f}"] = float((w * e.abs()).sum() / w.sum())  # CRPS of a point forecast is its absolute error
    return out


def _setup(config, ckpt, mean, noise, k_default=5):
    from hycom_emulator import rollout_rea as rr

    m = rr.load_model(config, ckpt)
    dev = m.module.device
    A = torch.load(mean, map_location=dev, weights_only=False)
    N = torch.load(noise, map_location="cpu", weights_only=False) if noise else None
    lay = _layout(m, A["k"])
    valid = torch.as_tensor(lay.valid(), device=dev)
    free = ~torch.as_tensor(np.asarray(m.datastore.boundary_mask.values, bool), device=dev) & torch.as_tensor(lay.nreal > 0, device=dev)
    return m, dev, A, N, valid, free


def calibrate(config: Path, ckpt: Path, mean: Path, noise: Path, starts: list[str], horizon: int, M: int, sweeps: int,
              batch: int, seed: int = 0) -> dict:
    """Run at T = 1 on held-out starts; per lead bin, ratio = mean over fields of spread / RMSE of the ensemble mean
    (pooled as root mean squares over starts and leads); T_b = 1 / ratio^2 (spread is linear in sqrt T). Writes T into
    noise.pt."""
    from hycom_emulator import rollout_rea as rr

    m, dev, A, N, valid, free = _setup(config, ckpt, mean, noise)
    s = Standardizer(m.datastore, dev)
    samplers = [Sampler(N, jb, m.shape, free, valid, dev) for jb in range(len(LEAD_BINS))]
    gen = torch.Generator().manual_seed(seed)
    acc = np.zeros((len(LEAD_BINS), len(FIELDS), 2))
    todo = _starts(m, starts, horizon)
    for b0 in range(0, len(todo), batch):
        for lead, pos, step in rr.unroll(m.module, m.window, m.times, todo[b0:b0 + batch], horizon):
            jb = lead_bin(lead)
            xh = s.z(step.pred)
            mu = xh + apply(A["A"][jb], xh, m.shape, A["k"])
            for i in range(xh.shape[0]):
                r = samplers[jb].sample(M, sweeps, gen, T=1.0)
                st = ens_stats(s.raw(mu[i][None] + r), step.truth[i], m)
                for a, f in enumerate(FIELDS):
                    acc[jb, a] += (st[f"spread_{f}"] ** 2, st[f"rmse_{f}"] ** 2)
        rr.release_pack_pages(m.datastore)
        print(f"calibrate: starts {b0 + 1}..{min(b0 + batch, len(todo))} of {len(todo)}", flush=True)
    ratio = np.sqrt(acc[..., 0] / acc[..., 1]) * np.sqrt((M + 1) / M)  # spread over error with the finite-M factor
    T = 1.0 / ratio.mean(1) ** 2
    d = torch.load(noise, map_location="cpu", weights_only=False)
    d["T"] = torch.as_tensor(T, dtype=torch.float32)
    d["calibration"] = {"starts": [str(t.date) for t in todo], "M": M, "sweeps": sweeps, "ratio_at_T1": ratio.tolist(), "T": T.tolist()}
    torch.save(d, noise)
    return d["calibration"]


def evaluate(config: Path, ckpt: Path, mean: Path, noise: Path, out: Path, starts: list[str], horizon: int, M: int,
             sweeps: int, batch: int, energy: Path | None, band_json: Path | None, seed: int = 0) -> None:
    """results.tsv: per start, lead and product (base: the free run; meancorr: x_hat + A phi(x_hat); ens: the members),
    RMSE (of the ensemble mean for ens), spread (ens), CRPS, and for ens and base the member KE ratio and, with an
    energy, the share of members inside the realism band per group."""
    from hycom_emulator import rollout_rea as rr
    from hycom_emulator.realism import Energy, _region

    m, dev, A, N, valid, free = _setup(config, ckpt, mean, noise)
    s = Standardizer(m.datastore, dev)
    samplers = [Sampler(N, jb, m.shape, free, valid, dev) for jb in range(len(LEAD_BINS))]
    gen = torch.Generator().manual_seed(seed)
    E = bnd = None
    if energy is not None:
        d = torch.load(energy, map_location="cpu", weights_only=False)
        E = Energy(d, _region(m.datastore), m.meta.level_ocean.values, [float(z) for z in m.meta.level.values], dev)
        bnd = json.loads(Path(band_json).read_text())
    rows = []
    todo = _starts(m, starts, horizon)
    for b0 in range(0, len(todo), batch):
        for lead, pos, step in rr.unroll(m.module, m.window, m.times, todo[b0:b0 + batch], horizon):
            jb = lead_bin(lead)
            xh = s.z(step.pred)
            mu = xh + apply(A["A"][jb], xh, m.shape, A["k"])
            for i, p in enumerate(pos):
                base = point_stats(step.pred[i], step.truth[i], m)
                mc = point_stats(s.raw(mu[i]), step.truth[i], m)
                r = samplers[jb].sample(M, sweeps, gen)
                mem_z = mu[i][None] + r
                mem = s.raw(mem_z)
                ens = ens_stats(mem, step.truth[i], m)

                def kef(x):
                    from hycom_emulator.rollout_rea import Step
                    st = Step(x, step.truth[i][None].expand(x.shape[0], -1, -1), step.x0[i][None].expand(x.shape[0], -1, -1),
                              step.pred_prev[i][None].expand(x.shape[0], -1, -1), step.truth_prev[i][None].expand(x.shape[0], -1, -1))
                    return rr.ke(st, m.geo, "pred") / rr.ke(st, m.geo, "truth")

                extra = {"ke_ratio_base": float(kef(step.pred[i][None])[0]), "ke_ratio_members": float(kef(mem).mean()),
                         "ke_ratio_meancorr": float(kef(s.raw(mu[i])[None])[0])}
                if E is not None:
                    for name, x in (("members", mem_z), ("base", xh[i][None])):
                        for g, v in E(x).items():
                            le = torch.log(v)
                            extra[f"inside_{g}_{name}"] = float(((le >= bnd["log_lo"][g]) & (le <= bnd["log_hi"][g])).float().mean())
                            extra[f"below_{g}_{name}"] = float((le < bnd["log_lo"][g]).float().mean())
                d0 = str(todo[b0 + p].date)
                for prod, st in (("base", base), ("meancorr", mc), ("ens", ens)):
                    rows.append({"start": d0, "lead": lead, "product": prod, **st, **(extra if prod != "meancorr" else {})})
        rr.release_pack_pages(m.datastore)
        print(f"evaluate: starts {b0 + 1}..{min(b0 + batch, len(todo))} of {len(todo)}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k_ for r in rows for k_ in r}, key=lambda k_: (k_ not in ("start", "lead", "product"), k_))
    with (out / "results.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys, delimiter="\t", lineterminator="\n", restval="")
        w.writeheader()
        w.writerows(rows)
    (out / "evaluate.json").write_text(json.dumps({"ckpt": str(ckpt), "mean": str(mean), "noise": str(noise), "M": M,
                                                   "sweeps": sweeps, "starts": [str(t.date) for t in todo], "horizon": horizon,
                                                   "T": N["T"].tolist(), "seed": seed}, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("config", type=Path)
    c.add_argument("ckpt", type=Path)
    c.add_argument("out", type=Path)
    c.add_argument("--stage", choices=("mean", "resid"), required=True)
    c.add_argument("--mean", type=Path)
    for q in (c,):
        q.add_argument("--starts", required=True)
        q.add_argument("--horizon", type=int, default=21)
        q.add_argument("--k", type=int, default=5)
        q.add_argument("--batch", type=int, default=8)
    fm = sub.add_parser("fit-mean")
    fm.add_argument("train", type=Path)
    fm.add_argument("held", type=Path)
    fm.add_argument("out", type=Path)
    fm.add_argument("--lambdas", default="1e-4,1e-3,1e-2,1e-1,1")
    fn = sub.add_parser("fit-noise")
    fn.add_argument("train", type=Path)
    fn.add_argument("held", type=Path)
    fn.add_argument("mean", type=Path)
    fn.add_argument("out", type=Path)
    fn.add_argument("--lambda", dest="lam", type=float, default=1e-3)
    for name in ("calibrate", "evaluate"):
        q = sub.add_parser(name)
        q.add_argument("config", type=Path)
        q.add_argument("ckpt", type=Path)
        q.add_argument("mean", type=Path)
        q.add_argument("noise", type=Path)
        if name == "evaluate":
            q.add_argument("out", type=Path)
            q.add_argument("--energy", type=Path)
            q.add_argument("--band", type=Path)
        q.add_argument("--starts", required=True)
        q.add_argument("--horizon", type=int, default=21)
        q.add_argument("--members", type=int, default=16)
        q.add_argument("--sweeps", type=int, default=10)
        q.add_argument("--batch", type=int, default=4)
        q.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    st = a.starts.split(",") if hasattr(a, "starts") else None
    if a.cmd == "collect":
        collect(a.config, a.ckpt, a.out, a.stage, st, a.horizon, a.k, a.batch, a.mean)
    elif a.cmd == "fit-mean":
        fit_mean(a.train, a.held, a.out, [float(x) for x in a.lambdas.split(",")])
    elif a.cmd == "fit-noise":
        fit_noise(a.train, a.held, a.mean, a.out, a.lam)
    elif a.cmd == "calibrate":
        print(json.dumps(calibrate(a.config, a.ckpt, a.mean, a.noise, st, a.horizon, a.members, a.sweeps, a.batch, a.seed), indent=1))
    else:
        evaluate(a.config, a.ckpt, a.mean, a.noise, a.out, st, a.horizon, a.members, a.sweeps, a.batch, a.energy, a.band, a.seed)


if __name__ == "__main__":
    main()
