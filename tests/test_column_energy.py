import itertools

import numpy as np
import torch

from hycom_emulator.column_energy import (ColumnEnergy, bin_medians, fair_crps, fit_edges, gaussian_crps, gibbs,
                                          to_bins, train_energy)


def planted(seed=0, n=3, q=3, scale=1.5):
    """A random 3-cell energy and its exact joint by enumeration."""
    torch.manual_seed(seed)
    model = ColumnEnergy(n, q)
    with torch.no_grad():
        w = scale * torch.randn(n * q, n * q)
        model.W.copy_(w + w.T)
        model.B.copy_(torch.randn(n, q))
    states = torch.tensor(list(itertools.product(range(q), repeat=n)))
    with torch.no_grad():
        p = torch.softmax(-model.energy(states), 0)
    return model, states, p


def test_pseudolikelihood_recovers_planted_joint():
    model, states, p = planted()
    rng = np.random.default_rng(1)
    x = states[rng.choice(len(states), 40000, p=p.numpy())].numpy()
    fit, log = train_energy(x, 3, steps=600, batch=4096, lr=0.05, l1=0.0, val_frac=0.0)
    with torch.no_grad():
        pf = torch.softmax(-fit.energy(states), 0)
    assert 0.5 * (pf - p).abs().sum() < 0.03
    assert log[-1]["train_pl"] < log[0]["train_pl"] + 1e-6


def test_gibbs_respects_clamps_and_samples_the_conditional():
    model, states, p = planted(seed=2)
    x0 = torch.full((1, 3), 2)
    clamp = torch.tensor([True, False, False])
    d = gibbs(model, x0, clamp, chains=4000, burn=30, keep=1, thin=1, seed=3)
    assert d.shape == (1, 4000, 3)
    assert (d[..., 0] == 2).all()
    sel = states[:, 0] == 2
    exact = (p[sel] / p[sel].sum()).numpy()
    emp = np.array([((d[0, :, 1] == s[1]) & (d[0, :, 2] == s[2])).float().mean().item() for s in states[sel]])
    assert np.abs(emp - exact).max() < 0.03


def test_symmetric_coupling_and_zero_self_blocks():
    m = ColumnEnergy(4, 5)
    with torch.no_grad():
        m.W.normal_()
    k = m.kmat()
    assert torch.allclose(k, k.T)
    for i in range(4):
        assert (k[i * 5:(i + 1) * 5, i * 5:(i + 1) * 5] == 0).all()


def test_logits_match_energy_differences():
    model, _, _ = planted(seed=4, n=4, q=3)
    x = torch.tensor([[0, 1, 2, 1]])
    with torch.no_grad():
        lg = model.logits(x)[0].numpy()
    for i in range(4):
        e = []
        for v in range(3):
            y = x.clone()
            y[0, i] = v
            e.append(model.energy(y).item())
        assert np.allclose(lg[i] - lg[i].mean(), -(np.array(e) - np.mean(e)), atol=1e-4)


def test_gaussian_crps_against_monte_carlo():
    rng = np.random.default_rng(0)
    mu, sd, y = 1.3, 0.7, 2.1
    x = rng.normal(mu, sd, (2, 400000))
    mc = np.abs(x[0] - y).mean() - 0.5 * np.abs(x[0] - x[1]).mean()
    assert abs(gaussian_crps(np.array(mu), np.array(sd), np.array(y)) - mc) < 3e-3


def test_fair_crps_matches_pairwise_definition():
    rng = np.random.default_rng(1)
    ens, y = rng.normal(size=(5, 7, 3)), rng.normal(size=(5, 3))
    m = ens.shape[1]
    pair = np.abs(ens[:, :, None] - ens[:, None, :]).sum((1, 2))
    ref = np.abs(ens - y[:, None]).mean(1) - pair / (2 * m * (m - 1))
    assert np.allclose(fair_crps(ens, y), ref)


def test_quantile_bins_and_medians():
    rng = np.random.default_rng(2)
    v = np.concatenate([rng.normal(size=(1000, 1)), np.zeros((1000, 1))], 1).astype(np.float32)
    v[:500, 1] = 1.0  # two distinct values only: most bins empty
    e = fit_edges(v, 8)
    b = to_bins(v, e)
    assert b.min() >= 0 and b.max() <= 7
    assert np.bincount(b[:, 0], minlength=8).min() > 100
    med = bin_medians(v, b, 8)
    assert np.all(np.diff(med[0]) > 0)
    assert np.isfinite(med).all()
    assert set(np.unique(np.take_along_axis(med, b.T, 1)[1])) == {0.0, 1.0}
