"""evaluate_gauss: calibration scores of a Gaussian forecast and the post-hoc factor. Synthetic arrays."""

import numpy as np
import torch

from hycom_emulator.evaluate_gauss import accumulate, calibration, summarize
from hycom_emulator.gauss import wcrps_gauss

NAMES = [f"{v}_{d}m" for v in ("temp", "salin", "u", "v") for d in (0, 10)] + ["ssh"]
F = len(NAMES)
LEADS, GRID = 2, 20000


def draw(seed, sigma_factor=1.0):
    """Truth drawn from N(mean, sigma^2) with sigma growing with lead; the forecast states sigma * sigma_factor."""
    rng = np.random.default_rng(seed)
    sigma = np.linspace(0.5, 2.0, F) * np.array([1.0, 2.0])[:, None, None] * np.ones((LEADS, GRID, F))
    mean = rng.normal(size=(LEADS, GRID, F))
    return rng.normal(size=(GRID, F)), mean + rng.normal(size=mean.shape) * sigma, mean, sigma * sigma_factor


def test_a_right_gaussian_has_spread_skill_one_and_the_normal_coverage():
    x0, truth, mean, sigma = draw(0)
    acc = {}
    accumulate(acc, x0, truth, mean, sigma, np.ones((GRID, F)))
    out = summarize(acc, NAMES, np.array([5.0, 10.0]))
    for name in ("temp_0m", "ssh", "temp"):
        for lead in ("1", "2"):
            s = out[name][lead]
            assert abs(s["spread_skill"] - 1) < 0.03 and abs(s["cover1"] - 0.683) < 0.01 and abs(s["cover2"] - 0.954) < 0.005, (name, lead, s)
    assert "sigma_cal" not in out["ssh"]["1"]


def test_crps_is_the_training_loss_per_point():
    _, truth, mean, sigma = draw(1)
    acc = {}
    accumulate(acc, truth[0, :50], truth[:, :50], mean[:, :50], sigma[:, :50], np.eye(50, 1).repeat(F, 1))
    theirs = wcrps_gauss(torch.tensor(mean[:, :1]), torch.tensor(truth[:, :1]), torch.tensor(sigma[:, :1]), average_grid=False, sum_vars=False,
                         weight=torch.ones(F, dtype=torch.float64))
    assert np.allclose(acc["crps"], theirs[:, 0].numpy())


def test_a_factor_fitted_on_one_split_calibrates_another():
    w, dz = np.ones((GRID, F)), np.array([5.0, 10.0])
    fit = {}
    accumulate(fit, *draw(2, sigma_factor=0.5), w)
    scale = calibration(fit)
    assert scale.shape == (LEADS, F) and np.allclose(scale, 2.0, rtol=0.03)
    acc = {}
    accumulate(acc, *draw(3, sigma_factor=0.5), w, scale)
    for lead in ("1", "2"):
        s = summarize(acc, NAMES, dz)["temp"][lead]
        assert abs(s["spread_skill"] - 0.5) < 0.02 and abs(s["spread_skill_cal"] - 1) < 0.04
        assert abs(s["cover1_cal"] - 0.683) < 0.015 and s["crps_cal"] < s["crps"]


def test_a_channel_without_points_gets_factor_one():
    x0, truth, mean, sigma = draw(4)
    w = np.ones((GRID, F))
    w[:, 3] = 0
    acc = {}
    accumulate(acc, x0, truth, mean, sigma, w)
    assert np.all(calibration(acc)[:, 3] == 1)
