"""Persistence scored with neural-lam's own loss: the baseline a rollout must beat at each lead.

    python -m hycom_emulator.persistence <nlam.yaml> <out.json> [--split test] [--ar-steps 4] [--loss wmse]

persistence_losses computes the loss as ForecasterModule does: standardized states, per_var_std = diff_std /
sqrt(feature weight), interior mask, one sample in memory at a time. Row i is sample i and column k the loss at
lead k + 1, so the column means are what neural-lam logs as <split>_loss_unroll<k + 1> for a forecast that keeps
the initial state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def persistence_losses(config_path: str, split: str, ar_steps: int, loss_name: str = "wmse", n: int | None = None) -> np.ndarray:
    import torch
    from neural_lam import metrics
    from neural_lam.config import load_config_and_datastore
    from neural_lam.loss_weighting import get_state_feature_weighting
    from neural_lam.weather_dataset import WeatherDataset

    import hycom_emulator.datastore  # noqa: F401  registers the hycom kind (and hycom_wmse's settings)
    from hycom_emulator.physics import hycom_wmse

    config, ds = load_config_and_datastore(config_path=config_path)
    stats = ds.get_standardization_dataarray("state")
    t = lambda name: torch.tensor(stats[name].values, dtype=torch.float32)  # noqa: E731
    mean, std, diff_std = t("state_mean"), t("state_std"), t("state_diff_std_standardized")
    w = torch.tensor(get_state_feature_weighting(config=config, datastore=ds), dtype=torch.float32)
    per_var_std = diff_std / torch.sqrt(w)
    mask = torch.tensor(~ds.boundary_mask.values.astype(bool))
    loss = {"wmse": metrics.wmse, "hycom_wmse": hycom_wmse}[loss_name]
    data = WeatherDataset(ds, split=split, ar_steps=ar_steps, num_past_forcing_steps=1, num_future_forcing_steps=1)
    idx = np.arange(len(data)) if n is None else np.linspace(0, len(data) - 1, min(n, len(data))).astype(int)
    rows = []
    for i in idx:
        init, target, _, _ = data[i]
        x0, target = (init[1] - mean) / std, (target - mean) / std
        rows.append([loss(x0[None], target[k][None], per_var_std, mask=mask).item() for k in range(ar_steps)])
    return np.array(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("out", type=Path)
    p.add_argument("--split", default="test")
    p.add_argument("--ar-steps", type=int, default=4)
    p.add_argument("--loss", default="wmse")
    a = p.parse_args()
    s = persistence_losses(a.config, a.split, a.ar_steps, a.loss)
    result = {
        "split": a.split,
        "loss": a.loss,
        "samples": len(s),
        "per_lead": {str(k + 1): float(s[:, k].mean()) for k in range(a.ar_steps)},
        "mean_over_leads": float(s.mean()),
    }
    a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
