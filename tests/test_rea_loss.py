"""rea_loss: rea_wmse's domain-mean term on a synthetic Gulf grid with land, a non-Gulf region, a band and fill levels."""

import numpy as np
import pytest
import torch
import xarray as xr
from neural_lam import metrics

from hycom_emulator import rea_loss
from hycom_emulator.conservation import apply, functionals
from hycom_emulator.evaluate_rea import cell_area, regions
from hycom_emulator.rea_loss import QUANTITIES, DomainMeans, rea_wmse, scales_from_series

LEVELS = np.array([0.0, 10.0, 100.0, 2000.0])
NAMES = [f"{v}_{d:g}m" for v in ("temp", "salin", "u", "v") for d in LEVELS] + ["ssh", "ubaro", "vbaro"]
NX, NY = 6, 5
N, F = NX * NY, len(NAMES)
SERIES_ORDER = ["ssh_mean", "ubaro_mean", "vbaro_mean", "temp_mean", "heat_content", "salin_mean", "salt_content", "u_mean", "v_mean"]


def _meta(rng):
    """Land at points 0-2, band on the last grid row and on land, Gulf on points < 20, fill below 100 m at 3-9."""
    ocean = np.ones(N, bool)
    ocean[:3] = False
    band = ~ocean | (np.arange(N) % NY == NY - 1)
    gulf = np.arange(N) < 20
    level_ocean = np.ones((N, LEVELS.size), bool)
    level_ocean[:3] = False
    level_ocean[3:10, 3] = False
    lat = np.repeat(np.linspace(18.0, 30.0, NX), NY)
    static = np.stack([lat, ocean, gulf], 1).astype(float)
    return xr.Dataset(
        {
            "static": (("grid_index", "static_feature"), static),
            "boundary_mask": ("grid_index", band.astype(np.int8)),
            "level_ocean": (("grid_index", "level"), level_ocean.astype(np.int8)),
            "state_std": ("state_feature", rng.uniform(0.5, 3.0, F)),
            "state_mean": ("state_feature", rng.normal(size=F)),
        },
        coords={"state_feature": NAMES, "static_feature": ["lat", "ocean", "gulf"], "level": LEVELS},
    )


def _batch(rng, b=2, t=3):
    pred = torch.tensor(rng.normal(size=(b, t, N, F)), dtype=torch.float32)
    target = torch.tensor(rng.normal(size=(b, t, N, F)), dtype=torch.float32)
    pred_std = torch.tensor(rng.uniform(0.2, 1.0, F), dtype=torch.float32)
    return pred, target, pred_std


def _interior(meta):
    return torch.tensor(~meta.boundary_mask.values.astype(bool))


def _direct_term(meta, pred, target, scales):
    """sum_q (Q_q(destandardized pred) - Q_q(destandardized target))^2 / s_q^2, with conservation.apply."""
    fs = functionals(NAMES, cell_area(meta), regions(meta)["gulf"], meta.level_ocean.values.astype(bool), LEVELS)
    fs = [next(f for f in fs if f.name == q) for q in QUANTITIES]
    phys = lambda x: x.double().numpy() * meta.state_std.values + meta.state_mean.values  # noqa: E731
    dq = apply(fs, phys(pred)) - apply(fs, phys(target))
    return ((dq / scales) ** 2).sum(-1)


def test_term_a_is_wmse_plus_lambda_times_the_direct_domain_mean_term():
    rng = np.random.default_rng(0)
    meta = _meta(rng)
    pred, target, pred_std = _batch(rng)
    scales = rng.uniform(0.05, 0.5, len(QUANTITIES))
    rea_loss.configure(DomainMeans.from_meta(meta), 1 / scales, mean_penalty=0.3)
    mask = _interior(meta)
    got = rea_wmse(pred, target, pred_std, mask=mask) - metrics.wmse(pred, target, pred_std, mask=mask)
    assert got.shape == (2, 3)
    assert np.allclose(got.numpy(), 0.3 * _direct_term(meta, pred, target, scales), rtol=1e-4)


def test_lambda_zero_is_exactly_wmse():
    rng = np.random.default_rng(1)
    meta = _meta(rng)
    pred, target, pred_std = _batch(rng)
    rea_loss.configure(DomainMeans.from_meta(meta), np.ones(len(QUANTITIES)), mean_penalty=0.0)
    mask = _interior(meta)
    assert torch.equal(rea_wmse(pred, target, pred_std, mask=mask), metrics.wmse(pred, target, pred_std, mask=mask))
    rea_loss.configure(None, None, mean_penalty=0.0)
    assert torch.equal(rea_wmse(pred, target, pred_std, mask=mask), metrics.wmse(pred, target, pred_std, mask=mask))


def test_per_variable_and_per_point_calls_return_wmse():
    rng = np.random.default_rng(2)
    meta = _meta(rng)
    pred, target, pred_std = _batch(rng)
    rea_loss.configure(DomainMeans.from_meta(meta), np.ones(len(QUANTITIES)), mean_penalty=5.0)
    mask = _interior(meta)
    for kw in ({"sum_vars": False}, {"average_grid": False}, {"sum_vars": False, "average_grid": False}):
        assert torch.equal(rea_wmse(pred, target, pred_std, mask=mask, **kw), metrics.wmse(pred, target, pred_std, mask=mask, **kw))


def test_domain_means_ignore_errors_outside_the_gulf_and_below_the_floor():
    rng = np.random.default_rng(3)
    meta = _meta(rng)
    means = DomainMeans.from_meta(meta)
    pred, target, _ = _batch(rng)
    noisy = pred.clone()
    outside = ~regions(meta)["gulf"]
    noisy[..., torch.tensor(outside), :] += 1e3
    noisy[..., 3:10, NAMES.index("salin_2000m")] += 1e3
    assert torch.allclose(means.errors(noisy, target), means.errors(pred, target), atol=1e-4)
    noisy[..., 12, NAMES.index("salin_2000m")] += 1.0
    assert not torch.allclose(means.errors(noisy, target), means.errors(pred, target), atol=1e-4)
    assert means.names == QUANTITIES and means.errors(pred, target).shape == (2, 3, len(QUANTITIES))


def test_a_uniform_offset_shows_up_as_itself_in_si_units():
    rng = np.random.default_rng(4)
    meta = _meta(rng)
    means = DomainMeans.from_meta(meta)
    target = torch.zeros(1, 1, N, F)
    pred = target.clone()
    pred[..., NAMES.index("ssh")] += 0.02 / float(meta.state_std.values[NAMES.index("ssh")])
    e = means.errors(pred, target)[0, 0]
    assert np.isclose(float(e[QUANTITIES.index("ssh_mean")]), 0.02, rtol=1e-5)
    assert torch.count_nonzero(e) == 1


def test_scales_skip_filled_rows_and_rows_after_the_train_end(tmp_path):
    rng = np.random.default_rng(5)
    time = np.datetime64("2021-12-25T12:00", "ns") + np.arange(10) * np.timedelta64(1, "D")
    filled = np.zeros(10, bool)
    filled[3] = True
    values = rng.normal(size=(10, 2, len(SERIES_ORDER)))
    np.savez(tmp_path / "s.npz", time=time, time_filled=filled, values=values,
             regions=np.array(["interior", "gulf"]), quantities=np.array(SERIES_ORDER))
    got = scales_from_series(tmp_path / "s.npz", "2021-12-31")
    later = [1, 2, 5, 6]  # 3 and 4 touch the filled row, 7-9 end after 2021-12-31; 6 is 2021-12-31T12
    cols = [SERIES_ORDER.index(q) for q in QUANTITIES]
    d = values[later, 1][:, cols] - values[[i - 1 for i in later], 1][:, cols]
    assert got.shape == (len(QUANTITIES),) and np.allclose(got, d.std(0))


def test_unconfigured_rea_wmse_raises(monkeypatch):
    monkeypatch.setattr(rea_loss, "_SETTINGS", None)
    rng = np.random.default_rng(6)
    with pytest.raises(RuntimeError, match="not configured"):
        rea_wmse(*_batch(rng))


def test_a_penalty_without_scales_is_refused():
    with pytest.raises(ValueError):
        rea_loss.configure(None, None, mean_penalty=0.1)


def test_registered_with_neural_lam():
    assert metrics.get_metric("rea_wmse") is rea_wmse
