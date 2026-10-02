"""Score a B00 checkpoint against persistence on one split, per field and lead.

Fields: T, S, total u and v (HYCOM archives hold baroclinic u-vel. plus u_btrop; mod_archiv.F90
writes u(:,:,k,n)), and SSH (srfhgt / g). T, S, u, v are weighted by the true layer thickness
times cell area; SSH by cell area. Cell area is cos^2(lat) up to a constant on this Mercator grid
(pscx = pscy = R cos(lat) dlon). Only interior ocean counts: boundary_mask = 0 excludes land and
the nest band, where neural-lam overwrites predictions with the truth.

For each field and lead:  rmse_model, rmse_persistence, rmse_persistence_inc (state + the IAU
share of increments added over the step; T, S, thknss only), and corr_change: the weighted
correlation between predicted and true change from the initial state. Persistence predicts no
change, so it has no corr_change. An ensemble datastore (stack_b00) is scored member by member:
the result then holds one score set and verdict per member under "members".

Run `python -m hycom_emulator.evaluate_b00 <nlam.yaml> <ckpt|none> <out.json> [--split test] [--ar-steps 2]`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

G = 9.806
FIELDS = ("temp", "salin", "u", "v", "ssh")


def _layers(names: list[str], var: str) -> np.ndarray:
    idx = [i for i, n in enumerate(names) if n.startswith(f"{var}_k")]
    return np.array(sorted(idx, key=lambda i: names[i]))


def field_values(x: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """(grid, feature) state -> per-field (grid, layer) arrays; SSH as (grid, 1) in metres."""
    col = {n: i for i, n in enumerate(names)}
    out = {v: x[:, _layers(names, v)] for v in ("temp", "salin")}
    out["u"] = x[:, _layers(names, "u")] + x[:, [col["ubaro"]]]
    out["v"] = x[:, _layers(names, "v")] + x[:, [col["vbaro"]]]
    out["ssh"] = x[:, [col["srfhgt"]]] / G
    return out


def weights(true: np.ndarray, names: list[str], area: np.ndarray) -> dict[str, np.ndarray]:
    thk = np.clip(true[:, _layers(names, "thknss")], 0.0, None)
    w = {f: thk * area[:, None] for f in ("temp", "salin", "u", "v")}
    w["ssh"] = area[:, None]
    return w


def rmse(a, b, w):
    return float(np.sqrt(np.sum(w * (a - b) ** 2) / np.sum(w)))


def wcorr(a, b, w):
    a = a - np.sum(w * a) / np.sum(w)
    b = b - np.sum(w * b) / np.sum(w)
    den = np.sqrt(np.sum(w * a * a) * np.sum(w * b * b))
    return float(np.sum(w * a * b) / den) if den > 0 else float("nan")


class Accumulator:
    """Pools squared errors and change moments over samples, per (field, lead)."""

    def __init__(self):
        self.sums: dict[tuple, dict[str, float]] = {}

    def add(self, key, true, init, w, preds):
        s = self.sums.setdefault(key, {"w": 0.0})
        s["w"] += float(np.sum(w))
        dt = true - init
        for name, p in preds.items():
            s[f"se_{name}"] = s.get(f"se_{name}", 0.0) + float(np.sum(w * (p - true) ** 2))
            if name == "model":
                dp = p - init
                for k, v in (("pp", dp * dp), ("tt", dt * dt), ("pt", dp * dt), ("p", dp), ("t", dt)):
                    s[k] = s.get(k, 0.0) + float(np.sum(w * v))

    def result(self):
        out = {}
        for (field, lead), s in self.sums.items():
            r = {k[3:]: float(np.sqrt(v / s["w"])) for k, v in s.items() if k.startswith("se_")}
            if "pp" in s:
                W = s["w"]
                cov = s["pt"] / W - (s["p"] / W) * (s["t"] / W)
                var_p = s["pp"] / W - (s["p"] / W) ** 2
                var_t = s["tt"] / W - (s["t"] / W) ** 2
                r["corr_change"] = float(cov / np.sqrt(var_p * var_t)) if var_p > 0 and var_t > 0 else float("nan")
            out.setdefault(field, {})[f"+{24 * lead}h"] = {f"rmse_{k}" if k != "corr_change" else k: v for k, v in r.items()}
        return out


def load(config_path: Path, ckpt: Path | None, split: str, ar_steps: int):
    """(datastore, WeatherDataset of the split, forecaster module or None)."""
    import torch
    from neural_lam.config import load_config_and_datastore
    from neural_lam.weather_dataset import WeatherDataset

    import hycom_emulator.datastore  # noqa: F401  registers the hycom kind
    import hycom_emulator.physics  # noqa: F401  registers hycom_graph_lam

    config, ds = load_config_and_datastore(config_path=str(config_path))
    data = WeatherDataset(ds, split=split, ar_steps=ar_steps, num_past_forcing_steps=1, num_future_forcing_steps=1)
    module = None
    if ckpt is not None:
        from neural_lam.train_model import load_forecaster_module_from_checkpoint

        device = "cuda" if torch.cuda.is_available() else "cpu"
        module = load_forecaster_module_from_checkpoint(str(ckpt), config, ds).to(device).eval()
    return ds, data, module


def evaluate(config_path: Path, ckpt: Path | None, split: str = "test", ar_steps: int = 2) -> dict:
    ds, data, module = load(config_path, ckpt, split, ar_steps)
    head = {"split": split, "ar_steps": ar_steps, "checkpoint": str(ckpt) if ckpt else None}
    if not ds.is_ensemble:
        return head | _score(ds, data, module, split, ar_steps, range(len(data)), None)
    members = ds.get_dataarray("state", split)["ensemble_member"].values.tolist()
    # WeatherDataset index = sample * n_members + member
    return head | {"members": {m: _score(ds, data, module, split, ar_steps, range(i, len(data), len(members)), m) for i, m in enumerate(members)}}


def _score(ds, data, module, split, ar_steps, indices, member) -> dict:
    names = ds.get_vars_names("state")
    interior = ~ds.boundary_mask.values.astype(bool)
    lat = ds.get_dataarray("static", None).sel(static_feature="lat").values
    area = np.cos(np.deg2rad(lat))[interior] ** 2
    forcing = ds.get_dataarray("forcing", split)
    if member is not None:
        forcing = forcing.sel(ensemble_member=member)
    inc_cols = increment_columns(ds)
    acc = Accumulator()
    for idx in indices:
        sample = data[idx]
        init_states, target_states, _, times = sample
        model_pred = forecast(module, sample) if module is not None else None
        x0 = init_states[-1].numpy()
        persist_inc = persistence_inc(x0, forcing, times, inc_cols)
        for lead in range(ar_steps):
            true = target_states[lead].numpy()
            preds = {"persistence": x0, "persistence_inc": persist_inc[lead]}
            if model_pred is not None:
                preds["model"] = model_pred[lead]
            fv_true = field_values(true[interior], names)
            fv_init = field_values(x0[interior], names)
            fv_pred = {k: field_values(p[interior], names) for k, p in preds.items()}
            w = weights(true[interior], names, area)
            for field in FIELDS:
                acc.add((field, lead + 1), fv_true[field], fv_init[field], w[field], {k: v[field] for k, v in fv_pred.items()})
    res = {"samples": len(indices), "interior_points": int(interior.sum()), "scores": acc.result()}
    if module is not None:
        res["verdict"] = verdict(res["scores"])
    return res


def forecast(module, sample) -> np.ndarray:
    """(ar_steps, grid, state_feature) model forecast in physical units for one WeatherDataset sample."""
    import torch

    batch = tuple(t[None].to(module.device) for t in sample)
    with torch.no_grad():
        batch = module.on_after_batch_transfer(batch, 0)
        pred, _, _, _ = module.common_step(batch)
    return (pred[0].cpu() * module.state_std.cpu() + module.state_mean.cpu()).numpy()


def persistence_inc(x0: np.ndarray, forcing, times, inc_cols: dict[int, int]) -> np.ndarray:
    """(ar_steps, grid, state_feature): x0 plus the IAU share of the increments accumulated by each lead."""
    out, x = [], x0.copy()
    for t in times:
        f_t = forcing.sel(time=np.datetime64(int(t), "ns")).values
        for i, j in inc_cols.items():
            x[:, i] += f_t[:, j]
        out.append(x.copy())
    return np.stack(out)


def increment_columns(ds) -> dict[int, int]:
    """State column -> forcing column of its increment (T, S, thickness layers)."""
    names, fnames = ds.get_vars_names("state"), ds.get_vars_names("forcing")
    return {i: fnames.index(f"inc_{n}") for i, n in enumerate(names) if f"inc_{n}" in fnames}


def verdict(scores: dict, partner_min: float = 0.0) -> dict:
    """Model RMSE below persistence + increments for every field and lead, and corr_change > partner_min.
    For a free run (zero increments) and for SSH, u and v this is plain persistence, card emu-b00-053's rule."""
    rows = {}
    for field, by_lead in scores.items():
        for lead, s in by_lead.items():
            ok = s.get("rmse_model", np.inf) < s["rmse_persistence_inc"] and s.get("corr_change", -1.0) > partner_min
            rows[f"{field} {lead}"] = bool(ok)
    return {"pass": all(rows.values()), "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("ckpt")
    p.add_argument("out", type=Path)
    p.add_argument("--split", default="test")
    p.add_argument("--ar-steps", type=int, default=2)
    a = p.parse_args()
    ckpt = None if a.ckpt == "none" else Path(a.ckpt)
    res = evaluate(a.config, ckpt, a.split, a.ar_steps)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
