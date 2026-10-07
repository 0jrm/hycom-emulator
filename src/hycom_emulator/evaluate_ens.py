"""Score a crps_graph_lam checkpoint as an ensemble, per field and lead, with a spread-skill check.

    python -m hycom_emulator.evaluate_ens <nlam.yaml> <checkpoint> <out.json> [--members 8] [--split test]
        [--ar-steps 10] [--limit n] [--region gulf]

For every state channel and lead: rmse_mean (RMSE of the ensemble mean), rmse_member (mean over members of
each member's MSE, as a root), spread (sqrt((M + 1) / M times the member variance), Fortin et al. 2014),
spread_skill = spread / rmse_mean, crps (fair, physical units) and rmse_persistence. A calibrated ensemble
has spread_skill near 1; below 1 it is overconfident. T, S, u and v are also scored as column aggregates
to 2000 m. Points and weights are evaluate_rea's. Same label: free forecasts given the true daily wind and
the true boundary band.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hycom_emulator.evaluate_rea import LEVEL_VARS, cell_area, level_weights, pack_path, point_weights, regions

SUMS = ("w", "e2_mean", "e2_member", "var", "crps", "e2_pers")


def fair_crps(ens: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """ens (M, ...) members, truth (...): mean_j |x_j - y| - sum_{j<k} |x_j - x_k| / (M (M - 1))."""
    m = ens.shape[0]
    coef = (2 * np.arange(m) - m + 1).reshape(m, *([1] * truth.ndim))
    return np.abs(ens - truth).mean(0) - (coef * np.sort(ens, axis=0)).sum(0) / (m * (m - 1))


def accumulate(acc: dict, x0: np.ndarray, truth: np.ndarray, ens: np.ndarray, w: np.ndarray) -> None:
    """Add one forecast: x0 (grid, feature), truth (lead, grid, feature), ens (member, lead, grid, feature)."""
    terms = {
        "e2_mean": (ens.mean(0) - truth) ** 2,
        "e2_member": ((ens - truth) ** 2).mean(0),
        "var": ens.var(0, ddof=1),
        "crps": fair_crps(ens, truth),
        "e2_pers": (x0[None] - truth) ** 2,
    }
    acc["w"] = acc.get("w", 0.0) + np.repeat(w.sum(0)[None], truth.shape[0], 0)
    for k, v in terms.items():
        acc[k] = acc.get(k, 0.0) + np.einsum("lgf,gf->lf", v, w)


def summarize(acc: dict, names: list[str], dz: np.ndarray, members: int) -> dict:
    groups = {n: [(j, 1.0)] for j, n in enumerate(names)}
    for var in LEVEL_VARS:
        groups[var] = [(j, dz[k]) for k, j in enumerate(j for j, n in enumerate(names) if n.rpartition("_")[0] == var)]
    out = {}
    for name, cols in groups.items():
        s = {k: sum(c * acc[k][:, j] for j, c in cols) / sum(c * acc["w"][:, j] for j, c in cols) for k in SUMS if k != "w"}
        rmse, spread = np.sqrt(s["e2_mean"]), np.sqrt(s["var"] * (members + 1) / members)
        out[name] = {
            str(lead + 1): {
                "rmse_mean": float(rmse[lead]),
                "rmse_member": float(np.sqrt(s["e2_member"][lead])),
                "spread": float(spread[lead]),
                "spread_skill": float(spread[lead] / rmse[lead]),
                "crps": float(s["crps"][lead]),
                "rmse_persistence": float(np.sqrt(s["e2_pers"][lead])),
            }
            for lead in range(len(rmse))
        }
    return out


def load(config: Path, ckpt: Path, split: str, ar_steps: int):
    """(datastore, WeatherDataset of the split, EnsembleForecasterModule on the GPU if there is one)."""
    import torch
    from neural_lam.config import load_config_and_datastore
    from neural_lam.models import MODELS, ARForecaster
    from neural_lam.train_model import build_predictor
    from neural_lam.weather_dataset import WeatherDataset

    import hycom_emulator.datastore  # noqa: F401  registers the hycom kind
    from hycom_emulator.ensemble import EnsembleForecasterModule

    nl_config, ds = load_config_and_datastore(config_path=str(config))
    data = WeatherDataset(ds, split=split, ar_steps=ar_steps, num_past_forcing_steps=1, num_future_forcing_steps=1)
    args = torch.load(ckpt, map_location="cpu", weights_only=False)["hyper_parameters"]["args"]
    forecaster = ARForecaster(build_predictor(MODELS[args.model], args, nl_config, ds), ds)
    module = EnsembleForecasterModule.load_from_checkpoint(str(ckpt), forecaster=forecaster, datastore=ds, weights_only=False, map_location="cpu")
    return ds, data, module.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def ensemble_forecast(module, sample, members: int) -> np.ndarray:
    """(member, lead, grid, state_feature) forecasts in physical units for one WeatherDataset sample."""
    import torch

    batch = tuple(t[None].to(module.device) for t in sample)
    with torch.no_grad():
        init, target, forcing, _ = module.on_after_batch_transfer(batch, 0)
        ens = module.forecast_members(init, forcing, target, members)[0]
    return (ens * module.state_std + module.state_mean).cpu().numpy()


def evaluate(config: Path, ckpt: Path, members: int, split: str, ar_steps: int, limit: int | None, region_name: str) -> dict:
    import xarray as xr

    ds, data, module = load(config, ckpt, split, ar_steps)
    meta = xr.open_zarr(pack_path(config) / "meta.zarr", consolidated=True).load()
    names = [str(n) for n in meta.state_feature.values]
    assert names == ds.get_vars_names("state"), "datastore and meta.zarr disagree on state features"
    levels = meta.level.values.astype(float)
    w = point_weights(names, cell_area(meta), regions(meta)[region_name], meta.level_ocean.values.astype(bool), levels)
    acc: dict = {}
    indices = range(len(data))[:limit]
    for i in indices:
        sample = data[i]
        accumulate(acc, sample[0][-1].numpy(), sample[1].numpy(), ensemble_forecast(module, sample, members), w)
    return {
        "label": "free forecast given the true daily wind and boundary band (no increments, no observations after t0)",
        "region": region_name, "split": split, "ar_steps": ar_steps, "members": members, "checkpoint": str(ckpt),
        "samples": len(indices), "fields": summarize(acc, names, level_weights(levels), members),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("ckpt", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--split", default="test")
    p.add_argument("--ar-steps", type=int, default=10)
    p.add_argument("--limit", type=int, default=None, help="score only the first n samples (smoke)")
    p.add_argument("--region", choices=("gulf", "interior"), default="gulf")
    a = p.parse_args()
    if a.members < 2:
        p.error("--members must be at least 2")
    r = evaluate(a.config, a.ckpt, a.members, a.split, a.ar_steps, a.limit, a.region)
    a.out.write_text(json.dumps(r, indent=1))
    for name in (*LEVEL_VARS, "ssh"):
        row = " ".join(f"{lead}d {s['spread_skill']:.2f}" for lead, s in r["fields"][name].items())
        print(f"{name:6s} spread/skill {row}")


if __name__ == "__main__":
    main()
