"""Score a Gaussian-head checkpoint (hycom_emulator.gauss) per field and lead: the mean, its predicted sigma and calibration.

    python -m hycom_emulator.evaluate_gauss <nlam.yaml> <checkpoint> <out.json> [--split test] [--ar-steps 10]
        [--limit n] [--region gulf] [--calibrate-on val]

For every state channel and lead: rmse_mean (RMSE of the predicted mean), rmse_persistence, sigma (rms of the
predicted std, physical units), spread_skill = sigma / rmse_mean, crps (Gaussian CRPS, physical units) and cover1,
cover2 (the weighted share of points whose truth lies within 1 and 2 sigma of the mean; 0.683 and 0.954 when the
forecast distribution is right). T, S, u and v are also scored as column aggregates to 2000 m. Points and weights are
evaluate_rea's; evaluate_rea scores the mean of the same checkpoint. Same label: free forecasts given the true daily
wind and the true boundary band.

--calibrate-on SPLIT fits a post-hoc factor per channel and lead on that split, c = rmse_mean / sigma, and scores
the scored split again with sigma times c: sigma_cal, spread_skill_cal, crps_cal, cover1_cal, cover2_cal. The fit
uses the same points, weights and --limit. c is the reference for what the network learned: a head whose sigma grows
with lead as the error does has c near 1 at every lead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hycom_emulator.evaluate_rea import LEVEL_VARS, cell_area, level_weights, pack_path, point_weights, regions
from hycom_emulator.gauss import gaussian_crps

def _sigma_terms(err: np.ndarray, mean: np.ndarray, sigma: np.ndarray, truth: np.ndarray, suffix: str) -> dict:
    crps = gaussian_crps(*map(torch.from_numpy, (mean, sigma, truth))).numpy()
    return {"var" + suffix: sigma**2, "crps" + suffix: crps, "in1" + suffix: err <= sigma, "in2" + suffix: err <= 2 * sigma}


def accumulate(acc: dict, x0: np.ndarray, truth: np.ndarray, mean: np.ndarray, sigma: np.ndarray, w: np.ndarray,
               scale: np.ndarray | None = None) -> None:
    """Add one forecast: x0 (grid, feature), truth, mean and sigma (lead, grid, feature), w (grid, feature); scale
    (lead, feature) also adds the scores of sigma * scale."""
    err = np.abs(mean - truth)
    terms = {"e2": err**2, "e2_pers": (x0[None] - truth) ** 2} | _sigma_terms(err, mean, sigma, truth, "")
    if scale is not None:
        terms |= _sigma_terms(err, mean, sigma * scale[:, None, :], truth, "_cal")
    acc["w"] = acc.get("w", 0.0) + np.repeat(w.sum(0)[None], truth.shape[0], 0)
    for k, v in terms.items():
        acc[k] = acc.get(k, 0.0) + np.einsum("lgf,gf->lf", v.astype(float), w)


def calibration(acc: dict) -> np.ndarray:
    """(lead, feature) rmse_mean / sigma of each channel; 1 where a channel has no scored point."""
    return np.sqrt(np.divide(acc["e2"], acc["var"], out=np.ones_like(acc["e2"]), where=acc["var"] > 0))


def summarize(acc: dict, names: list[str], dz: np.ndarray) -> dict:
    groups = {n: [(j, 1.0)] for j, n in enumerate(names)}
    for var in LEVEL_VARS:
        groups[var] = [(j, dz[k]) for k, j in enumerate(j for j, n in enumerate(names) if n.rpartition("_")[0] == var)]
    suffixes = ("", "_cal") if "var_cal" in acc else ("",)
    out = {}
    for name, cols in groups.items():
        s = {k: sum(c * acc[k][:, j] for j, c in cols) / sum(c * acc["w"][:, j] for j, c in cols) for k in acc if k != "w"}
        rmse = np.sqrt(s["e2"])
        out[name] = {}
        for lead in range(len(rmse)):
            row = {"rmse_mean": float(rmse[lead]), "rmse_persistence": float(np.sqrt(s["e2_pers"][lead]))}
            for x in suffixes:
                sigma = float(np.sqrt(s["var" + x][lead]))
                row |= {"sigma" + x: sigma, "spread_skill" + x: sigma / float(rmse[lead]), "crps" + x: float(s["crps" + x][lead]),
                        "cover1" + x: float(s["in1" + x][lead]), "cover2" + x: float(s["in2" + x][lead])}
            out[name][str(lead + 1)] = row
    return out


def gaussian_forecast(module, sample) -> tuple[np.ndarray, np.ndarray]:
    """(mean, sigma), each (lead, grid, state_feature) in physical units, for one WeatherDataset sample."""
    batch = tuple(t[None].to(module.device) for t in sample)
    with torch.no_grad():
        pred, _, sigma, _ = module.common_step(module.on_after_batch_transfer(batch, 0))
    if sigma is None:
        raise ValueError("the checkpoint has no Gaussian head (trained without --output_std)")
    std, mean = module.state_std.cpu(), module.state_mean.cpu()
    return (pred[0].cpu() * std + mean).numpy(), (sigma[0].cpu() * std).numpy()


def evaluate(config: Path, ckpt: Path, split: str, ar_steps: int, limit: int | None, region_name: str,
             calibrate_on: str | None = None) -> dict:
    import xarray as xr
    from neural_lam.weather_dataset import WeatherDataset

    from hycom_emulator.evaluate_b00 import load

    ds, data, module = load(config, ckpt, split, ar_steps)
    meta = xr.open_zarr(pack_path(config) / "meta.zarr", consolidated=True).load()
    names = [str(n) for n in meta.state_feature.values]
    assert names == ds.get_vars_names("state"), "datastore and meta.zarr disagree on state features"
    levels = meta.level.values.astype(float)
    w = point_weights(names, cell_area(meta), regions(meta)[region_name], meta.level_ocean.values.astype(bool), levels)

    def scored(dataset, scale=None) -> tuple[dict, int]:
        acc: dict = {}
        indices = range(len(dataset))[:limit]
        for i in indices:
            sample = dataset[i]
            accumulate(acc, sample[0][-1].numpy(), sample[1].numpy(), *gaussian_forecast(module, sample), w, scale)
        return acc, len(indices)

    out = {
        "label": "free forecast given the true daily wind and boundary band (no increments, no observations after t0)",
        "region": region_name, "split": split, "ar_steps": ar_steps, "checkpoint": str(ckpt),
    }
    scale = None
    if calibrate_on:
        fit, n = scored(WeatherDataset(ds, split=calibrate_on, ar_steps=ar_steps, num_past_forcing_steps=1, num_future_forcing_steps=1))
        scale = calibration(fit)
        out["calibration"] = {"split": calibrate_on, "samples": n,
                              "factor": {name: {str(k + 1): float(c) for k, c in enumerate(scale[:, j])} for j, name in enumerate(names)}}
    acc, out["samples"] = scored(data, scale)
    return out | {"fields": summarize(acc, names, level_weights(levels))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("ckpt", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--split", default="test")
    p.add_argument("--ar-steps", type=int, default=10)
    p.add_argument("--limit", type=int, default=None, help="score only the first n samples (smoke)")
    p.add_argument("--region", choices=("gulf", "interior"), default="gulf")
    p.add_argument("--calibrate-on", default=None, help="fit sigma's per-channel, per-lead factor on this split (val)")
    a = p.parse_args()
    r = evaluate(a.config, a.ckpt, a.split, a.ar_steps, a.limit, a.region, a.calibrate_on)
    a.out.write_text(json.dumps(r, indent=1))
    for name in (*LEVEL_VARS, "ssh"):
        row = " ".join(f"{lead}d {s['spread_skill']:.2f}/{s['cover1']:.2f}" for lead, s in r["fields"][name].items())
        print(f"{name:6s} spread/skill, 1-sigma cover {row}")


if __name__ == "__main__":
    main()
