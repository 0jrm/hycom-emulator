"""gauss: the weighted Gaussian CRPS, the head's init from a deterministic checkpoint and the routing. Synthetic pack."""

import math

import neural_lam.train_model as tm
import numpy as np
import pytest
import torch
from neural_lam import metrics
from neural_lam.models import ARForecaster, ForecasterModule
from neural_lam.models.step_predictors.graph.graph_lam import GraphLAM
from test_ensemble import HIDDEN, ds, rollout_batch  # noqa: F401  ds is a fixture

from hycom_emulator import nlam
from hycom_emulator.gauss import INIT_STD, GaussForecasterModule, module_factory, split_args, wcrps_gauss


def predictor(cls, datastore, seed=0, output_std=True):
    torch.manual_seed(seed)
    return cls(datastore=datastore, graph_name="multiscale", hidden_dim=HIDDEN, processor_layers=2, output_std=output_std)


def test_wcrps_is_the_closed_form_on_a_hand_example():
    # z = 0: 2 phi(0) - 1/sqrt(pi); z = 0.5, sigma 2: 2 (0.5 (2 Phi(0.5) - 1) + 2 phi(0.5) - 1/sqrt(pi))
    pred, target, sigma = torch.tensor([[0.0, 0.0]], dtype=torch.float64), torch.tensor([[0.0, 1.0]], dtype=torch.float64), torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    entry = wcrps_gauss(pred, target, sigma, average_grid=False, sum_vars=False, weight=torch.tensor([1.0, 4.0], dtype=torch.float64))
    hand = [2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi), 2 * (0.5 * (2 * 0.6914624612740131 - 1) + 2 * 0.3520653267642995 - 1 / math.sqrt(math.pi)) / 4]
    assert torch.allclose(entry, torch.tensor([hand], dtype=torch.float64), rtol=1e-12)


def test_wcrps_reduces_like_neural_lams_metrics():
    g = torch.Generator().manual_seed(0)
    pred, target = torch.randn(2, 3, 7, 4, generator=g, dtype=torch.float64), torch.randn(2, 3, 7, 4, generator=g, dtype=torch.float64)
    sigma, weight = torch.rand(2, 3, 7, 4, generator=g, dtype=torch.float64) + 0.1, torch.rand(4, generator=g, dtype=torch.float64) + 0.5
    mask = torch.tensor([True, False, True, True, False, True, True])
    theirs = metrics.crps_gauss(pred, target, sigma, average_grid=False, sum_vars=False) / weight
    assert torch.allclose(wcrps_gauss(pred, target, sigma, mask=mask, weight=weight), theirs[..., mask, :].mean(-2).sum(-1), rtol=1e-12)
    assert torch.allclose(wcrps_gauss(pred, target, sigma, average_grid=False, weight=weight), theirs.sum(-1), rtol=1e-12)


def test_wcrps_tends_to_wmae_as_sigma_vanishes():
    g = torch.Generator().manual_seed(1)
    pred, target, weight = torch.randn(3, 9, 4, generator=g, dtype=torch.float64), torch.randn(3, 9, 4, generator=g, dtype=torch.float64), torch.rand(4, dtype=torch.float64) + 0.5
    assert torch.allclose(wcrps_gauss(pred, target, torch.full_like(pred, 1e-9), weight=weight), metrics.wmae(pred, target, weight), atol=1e-8)


def test_wcrps_falls_as_sigma_moves_to_the_true_error_scale():
    g = torch.Generator().manual_seed(2)
    scale = torch.tensor([0.3, 1.0, 3.0], dtype=torch.float64)
    target = torch.randn(50000, 3, generator=g, dtype=torch.float64) * scale
    losses = [wcrps_gauss(torch.zeros_like(target), target, (f * scale).expand_as(target), weight=torch.ones(3, dtype=torch.float64)).item()
              for f in (0.25, 0.5, 1.0, 2.0, 4.0)]
    assert losses[0] > losses[1] > losses[2] < losses[3] < losses[4], losses


def deterministic_parent(ds, tmp_path):
    config, datastore = ds
    parent = ForecasterModule(forecaster=ARForecaster(predictor(GraphLAM, datastore, output_std=False), datastore), config=config, datastore=datastore)
    torch.save({"state_dict": parent.state_dict()}, tmp_path / "parent.ckpt")
    return parent


def gauss_module(ds, tmp_path, cls=GraphLAM, forecaster=ARForecaster):
    config, datastore = ds
    parent = deterministic_parent(ds, tmp_path)
    module = module_factory(str(tmp_path / "parent.ckpt"))(forecaster=forecaster(predictor(cls, datastore, seed=7), datastore),
                                                           config=config, datastore=datastore, loss="wcrps_gauss")
    return parent, module


def test_init_from_reproduces_the_deterministic_mean_and_sets_the_std_bias(ds, tmp_path):
    parent, module = gauss_module(ds, tmp_path)
    init, target, forcing, _ = rollout_batch(ds[1])
    mean, sigma = module.forecaster(init, forcing, target)
    det, _ = parent.forecaster(init, forcing, target)
    # The same weights through a 2F-row matmul round differently in float32 than through the F-row one (~1e-6).
    assert torch.allclose(mean, det, rtol=0, atol=1e-5), "the mean rows are the checkpoint's"
    sigma0 = INIT_STD * module.forecaster.predictor.diff_std
    assert sigma.shape == det.shape and torch.allclose(sigma, sigma0.expand_as(sigma), rtol=1e-6)


def test_the_module_weighs_by_the_change_std_and_refuses_other_set_ups(ds, tmp_path):
    config, datastore = ds
    _, module = gauss_module(ds, tmp_path)
    plain = ForecasterModule(forecaster=ARForecaster(predictor(GraphLAM, datastore, output_std=False), datastore), config=config, datastore=datastore)
    assert torch.equal(module.per_var_std, plain.per_var_std), "the weight of wmse and afcrps"
    batch = rollout_batch(datastore)
    pred, target, sigma, loss = module._compute_prediction_and_loss(batch)
    expected = wcrps_gauss(pred, target, sigma, mask=module.interior_mask_bool, weight=plain.per_var_std).mean(0)
    assert loss.shape == (target.shape[1],) and torch.allclose(loss, expected)
    assert module.loss(pred, target, sigma, average_grid=False).shape == target.shape[:-1], "the test pass's spatial maps"
    with pytest.raises(ValueError, match="output_std"):
        GaussForecasterModule(forecaster=ARForecaster(predictor(GraphLAM, datastore, output_std=False), datastore), config=config,
                              datastore=datastore, loss="wcrps_gauss")
    with pytest.raises(ValueError, match="wcrps_gauss"):
        GaussForecasterModule(forecaster=ARForecaster(predictor(GraphLAM, datastore), datastore), config=config, datastore=datastore, loss="nll")
    with pytest.raises(TypeError, match="weight"):
        ForecasterModule(forecaster=ARForecaster(predictor(GraphLAM, datastore), datastore), config=config, datastore=datastore,
                         loss="wcrps_gauss")._compute_prediction_and_loss(batch)


def test_split_args():
    init_from, rest = split_args(["--config_path", "n.yaml", "--model", "graph_lam", "--output_std", "--loss", "wcrps_gauss", "--init_from", "a.ckpt"])
    assert init_from == "a.ckpt" and rest == ["--config_path", "n.yaml", "--output_std", "--loss", "wcrps_gauss", "--model", "graph_lam"]


@pytest.mark.parametrize("argv, match", [
    (["--loss", "wcrps_gauss"], "--output_std"),
    (["--loss", "wcrps_gauss", "--init_from", "a.ckpt"], "--output_std"),
    (["--loss", "wcrps_gauss", "--output_std", "--model", "hi_lam"], "graph_lam"),
    (["--loss", "wcrps_gauss", "--output_std", "--log_domain_means"], "log_domain_means"),
])
def test_routing_refuses_an_inconsistent_set(argv, match, monkeypatch, capsys):
    monkeypatch.setattr(tm, "main", lambda rest: pytest.fail(f"reached neural-lam with {rest}"))
    monkeypatch.setattr(tm, "ForecasterModule", tm.ForecasterModule)
    monkeypatch.setattr(tm, "ARForecaster", tm.ARForecaster)
    with pytest.raises(SystemExit) as e:
        nlam.main(["nlam", "train_model", "--model", "graph_lam", *argv])
    assert match in str(e.value) + capsys.readouterr().err


def test_routing_installs_the_module(monkeypatch):
    calls = []
    monkeypatch.setattr(tm, "main", calls.append)
    monkeypatch.setattr(tm, "ForecasterModule", tm.ForecasterModule)
    monkeypatch.setattr(tm, "ARForecaster", tm.ARForecaster)
    nlam.main(["nlam", "train_model", "--model", "graph_lam", "--output_std", "--loss", "wcrps_gauss", "--init_from", "a.ckpt"])
    assert calls == [["--output_std", "--loss", "wcrps_gauss", "--model", "graph_lam"]]
    assert tm.ForecasterModule.__qualname__ == "module_factory.<locals>.build"
