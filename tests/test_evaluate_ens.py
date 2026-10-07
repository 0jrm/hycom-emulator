"""evaluate_ens: ensemble scores and the spread-skill ratio. Synthetic arrays."""

import numpy as np
import torch

from hycom_emulator.ensemble import crps_ensemble
from hycom_emulator.evaluate_ens import accumulate, fair_crps, summarize

NAMES = [f"{v}_{d}m" for v in ("temp", "salin", "u", "v") for d in (0, 10)] + ["ssh"]
F = len(NAMES)


def test_fair_crps_is_the_training_loss_per_point():
    rng = np.random.default_rng(0)
    ens, truth = rng.normal(size=(5, 3, 7, 2)), rng.normal(size=(3, 7, 2))
    ours = fair_crps(ens, truth)
    theirs = crps_ensemble(torch.tensor(ens).movedim(0, -3), torch.tensor(truth), torch.ones(2, dtype=torch.float64),
                           average_grid=False, sum_vars=False, alpha=1.0)
    assert np.allclose(ours, theirs.numpy())


def test_an_ensemble_drawn_like_the_truth_has_spread_skill_one():
    rng = np.random.default_rng(1)
    members, leads, grid = 4, 2, 20000
    scale = np.linspace(0.5, 2.0, F)
    truth = rng.normal(size=(leads, grid, F)) * scale
    ens = rng.normal(size=(members, leads, grid, F)) * scale
    acc = {}
    accumulate(acc, np.zeros((grid, F)), truth, ens, np.ones((grid, F)))
    out = summarize(acc, NAMES, np.array([5.0, 10.0]), members)
    for name in ("temp_0m", "ssh", "temp"):
        for lead in ("1", "2"):
            assert abs(out[name][lead]["spread_skill"] - 1) < 0.03, (name, lead, out[name][lead])


def test_members_on_the_truth_score_zero():
    rng = np.random.default_rng(2)
    truth = rng.normal(size=(2, 50, F))
    acc = {}
    accumulate(acc, rng.normal(size=(50, F)), truth, np.stack([truth, truth]), np.ones((50, F)))
    s = summarize(acc, NAMES, np.array([5.0, 10.0]), 2)["ssh"]["1"]
    assert s["rmse_mean"] == s["spread"] == s["crps"] == 0 and s["rmse_persistence"] > 0
