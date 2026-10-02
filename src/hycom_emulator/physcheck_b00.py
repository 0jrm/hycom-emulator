"""Physical consistency of B00 forecasts, judged against the truth's own residuals.

For every sample of a split (every member of an ensemble) and both leads, on interior points, the
same checks run on the truth, the model and persistence + increments:

  neg_thk_frac     fraction of (point, layer) thicknesses below -1 cm; neg_thk_min (m), the lowest
  col_err_m        RMS of the column thickness sum minus the truth's (m); HYCOM keeps the sum at the
                   bottom depth (within 1 mm in the 05.3 archives)
  inversion_frac   fraction of adjacent layer pairs, both thicker than 1 m, whose sigma2 is lower below
                   by more than 0.02 kg/m3 (HYCOM's 7-term sigma-2 polynomial)
  iso_spread       median over layers 16-41 of the interquartile range of sigma2 where the layer is
                   thicker than 5 m: how well isopycnal layers keep their density
  baro_resid       RMS of the thickness-weighted mean of the baroclinic velocity (m/s), p-point thickness
  lap_<field>      RMS of the 5-point Laplacian of the 24 h change of SSH, T k01, total u k01 and
                   thknss k20 (> truth: grid-scale noise added; < truth: smoothed)
  ke               mean surface kinetic energy (m2/s2)

Run `python -m hycom_emulator.physcheck_b00 <nlam.yaml> <ckpt> <out.json> [--split test]`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hycom_emulator.evaluate_b00 import load
from hycom_emulator.inspect_b00 import ONEM, Case, Forcing, Grid
from hycom_emulator.physics import layer_columns, sigma2

WHO = ("truth", "model", "pinc")


def checks(names, grid, inner, interior, x0, x, true) -> dict[str, float]:
    """Checks of state x (grid, state_feature) after a step from x0; true is the target."""
    col = {n: i for i, n in enumerate(names)}
    th, tc, sc, uc, vc = (layer_columns(names, v) for v in ("thknss", "temp", "salin", "u", "v"))
    xi, ti = x[interior], true[interior]
    dp, dpt = xi[:, th] / ONEM, ti[:, th] / ONEM
    w = np.clip(dp, 0, None)
    tot = np.maximum(w.sum(1), 1e-6)
    sig = sigma2(xi[:, tc], xi[:, sc])
    thick = (dp[:, :-1] > 1) & (dp[:, 1:] > 1)
    spread = [np.subtract(*np.percentile(sig[dp[:, k] > 5, k], [75, 25])) for k in range(15, len(th)) if (dp[:, k] > 5).sum() > 100]
    out = {
        "neg_thk_frac": float((dp < -0.01).mean()),
        "neg_thk_min": float(dp.min()),
        "col_err_m": float(np.sqrt(np.mean((dp.sum(1) - dpt.sum(1)) ** 2))),
        "inversion_frac": float(((sig[:, 1:] < sig[:, :-1] - 0.02) & thick).sum() / max(thick.sum(), 1)),
        "iso_spread": float(np.median(spread)) if spread else float("nan"),
        "baro_resid": float(np.sqrt(np.mean(((xi[:, uc] * w).sum(1) / tot) ** 2 + ((xi[:, vc] * w).sum(1) / tot) ** 2))),
    }
    fields = {"ssh": lambda a: a[:, col["srfhgt"]], "t01": lambda a: a[:, col["temp_k01"]],
              "u01": lambda a: a[:, col["u_k01"]] + a[:, col["ubaro"]], "thk20": lambda a: a[:, col["thknss_k20"]]}
    for name, f in fields.items():
        lap = _laplacian(grid.map(f(x) - f(x0)))
        out[f"lap_{name}"] = float(np.sqrt(np.nanmean(lap[inner] ** 2)))
    u, v = xi[:, col["u_k01"]] + xi[:, col["ubaro"]], xi[:, col["v_k01"]] + xi[:, col["vbaro"]]
    out["ke"] = float(np.mean(u**2 + v**2))
    return out


def _laplacian(a):
    return a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1]


def physcheck(config: Path, ckpt: Path, split: str = "test") -> dict:
    ds, data, module = load(config, ckpt, split, ar_steps=2)
    names = ds.get_vars_names("state")
    interior = ~ds.boundary_mask.values.astype(bool)
    grid = Grid(ds)
    m = grid.map(interior.astype(float))
    inner = (m[1:-1, 1:-1] == 1) & (m[:-2, 1:-1] == 1) & (m[2:, 1:-1] == 1) & (m[1:-1, :-2] == 1) & (m[1:-1, 2:] == 1)
    forcing_of = Forcing(ds, split)
    rows = []
    for idx in range(len(data)):
        for lead in (0, 1):
            c = Case(ds, data, module, forcing_of, idx, lead)
            r = {"member": c.member, "date": c.date, "lead_h": 24 * (lead + 1)}
            for who, x in zip(WHO, (c.true, c.model, c.pinc)):
                r[who] = checks(names, grid, inner, interior, c.x0, x, c.true)
            rows.append(r)
    return {"checkpoint": str(ckpt), "split": split, "summary": summarize(rows), "samples": rows}


def summarize(rows) -> dict:
    """Mean over samples per lead and source; neg_thk_min is the lowest."""
    out = {}
    for lead in sorted({r["lead_h"] for r in rows}):
        sel = [r for r in rows if r["lead_h"] == lead]
        for who in WHO:
            s = {k: float(np.nanmean([r[who][k] for r in sel])) for k in sel[0][who]}
            s["neg_thk_min"] = float(min(r[who]["neg_thk_min"] for r in sel))
            out[f"+{lead}h {who}"] = s
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("ckpt", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--split", default="test")
    a = p.parse_args()
    res = physcheck(a.config, a.ckpt, a.split)
    a.out.write_text(json.dumps(res, indent=1))
    keys = list(next(iter(res["summary"].values())))
    print("".ljust(13) + "".join(k[:12].rjust(13) for k in keys))
    for name, s in res["summary"].items():
        print(name.ljust(13) + "".join(f"{s[k]:13.4g}" for k in keys))


if __name__ == "__main__":
    main()
