"""physics: thickness projection, hycom_wmse and the ablation arms. Synthetic data, no RCC."""

import numpy as np
import pytest
import torch
import xarray as xr
from neural_lam import metrics

from hycom_emulator.physics import CONTEXT, ONEM, apply_arm, group_weights, hycom_wmse, layer_columns, project_thickness

L = 3
NAMES = [f"{v}_k{k:02d}" for v in ("temp", "salin", "thknss", "u", "v") for k in range(1, L + 1)] + ["srfhgt", "montg1", "ubaro", "vbaro"]
TH = layer_columns(NAMES, "thknss")
VAR = torch.tensor([1e-4, 1.0, 4.0]) * ONEM**2  # change variance: top layer near-fixed, bottom most variable


class _Stub:
    """The parts of a datastore CONTEXT.configure reads."""

    def __init__(self, physics=None):
        self.config = {"physics": physics} if physics else {}
        f = len(NAMES)
        mean = np.zeros(f, np.float32)
        std = np.ones(f, np.float32)
        mean[TH], std[TH] = 50 * ONEM, 30 * ONEM
        self.stats = xr.Dataset({"state_mean": ("f", mean), "state_std": ("f", std)})

    def get_vars_names(self, category):
        return NAMES

    def get_standardization_dataarray(self, category):
        return self.stats


def _state(rng, n=6, empty=()):
    """(1, n, feature) physical state: T, S plausible, thicknesses positive except `empty` layers."""
    x = rng.normal(size=(1, n, len(NAMES))).astype(np.float32)
    t, s = layer_columns(NAMES, "temp"), layer_columns(NAMES, "salin")
    x[..., t] = 20 - 5 * np.arange(L)
    x[..., s] = 36
    x[..., TH] = rng.uniform(10, 100, size=(1, n, L)) * ONEM
    for k in empty:
        x[..., TH[k]] = 0
    return x


def _std(x, stub):
    return torch.tensor((x - stub.stats.state_mean.values) / stub.stats.state_std.values)


def test_projection_keeps_columns_positive_and_closed():
    rng = np.random.default_rng(0)
    stub = _Stub()
    mean, std = (torch.tensor(stub.stats[v].values) for v in ("state_mean", "state_std"))
    prev = _std(_state(rng), stub)
    delta = torch.tensor(rng.normal(scale=3, size=prev.shape).astype(np.float32), requires_grad=True)
    raw = prev + delta
    out = project_thickness(raw, prev, mean, std, torch.tensor(TH), VAR)
    dp, dp0 = (a[..., TH] * std[TH] + mean[TH] for a in (out, prev))
    assert (dp >= 0).all()
    torch.testing.assert_close(dp.sum(-1), dp0.sum(-1), rtol=1e-5, atol=1e-2)
    other = [i for i in range(len(NAMES)) if i not in TH]
    torch.testing.assert_close(out[..., other], raw[..., other])
    out.sum().backward()
    assert torch.isfinite(delta.grad).all()


def test_projection_spreads_the_shortfall_by_change_variance():
    stub = _Stub()
    mean, std = (torch.tensor(stub.stats[v].values) for v in ("state_mean", "state_std"))
    prev = _state(np.random.default_rng(5))
    prev[..., TH] = np.array([1.0, 10.0, 89.0]) * ONEM  # column of 100 m
    new = prev.copy()
    new[..., TH] = np.array([1.0, 12.0, 80.0]) * ONEM  # 7 m short: the bottom layer absorbs it
    out = project_thickness(_std(new, stub), _std(prev, stub), mean, std, torch.tensor(TH), VAR)
    dp = (out[..., TH] * std[TH] + mean[TH]).numpy() / ONEM
    share = np.array([1e-4, 1.0, 4.0]) / 5.0001  # the 7 m shortfall, spread by change variance
    np.testing.assert_allclose(dp[0, 0], [1.0, 12.0, 80.0] + 7 * share, rtol=1e-5)
    assert abs(dp[0, 0, 0] - 1.0) < 1e-3  # the near-fixed top layer barely moves
    new[..., TH] = np.array([1.0, 12.0, 95.0]) * ONEM  # 8 m too deep
    out = project_thickness(_std(new, stub), _std(prev, stub), mean, std, torch.tensor(TH), VAR)
    np.testing.assert_allclose((out[..., TH] * std[TH] + mean[TH]).numpy()[0, 0] / ONEM, [1.0, 12.0, 95.0] - 8 * share, rtol=1e-5)



def test_projection_keeps_empty_bottom_layers_empty():
    stub = _Stub()
    mean, std = (torch.tensor(stub.stats[v].values) for v in ("state_mean", "state_std"))
    prev = _state(np.random.default_rng(6))
    prev[..., TH] = np.array([1.0, 99.0, 0.0]) * ONEM  # the deepest layer is empty
    new = prev.copy()
    new[..., TH] = np.array([1.0, 90.0, 0.5]) * ONEM
    out = project_thickness(_std(new, stub), _std(prev, stub), mean, std, torch.tensor(TH), VAR)
    dp = (out[..., TH] * std[TH] + mean[TH]).numpy()[0, 0] / ONEM
    share = np.array([1e-4, 1.0]) / 1.0001  # the 8.5 m shortfall goes to the layers with mass, by change variance
    np.testing.assert_allclose(dp, [1.0 + 8.5 * share[0], 90.0 + 8.5 * share[1], 0.5], rtol=1e-5)


def test_plain_settings_reduce_to_wmse():
    rng = np.random.default_rng(1)
    stub = _Stub()
    CONTEXT.configure(stub)
    target, pred = _std(_state(rng), stub), _std(_state(rng), stub)
    pred_std = torch.full((len(NAMES),), 0.7)
    mask = torch.tensor([True, True, False, True, True, True])
    for kw in ({}, {"average_grid": False, "sum_vars": False}):
        torch.testing.assert_close(hycom_wmse(pred, target, pred_std, mask=mask, **kw), metrics.wmse(pred, target, pred_std, mask=mask, **kw))


def test_empty_layers_weigh_nothing():
    rng = np.random.default_rng(2)
    stub = _Stub({"thickness_weighted": True})
    CONTEXT.configure(stub)
    target = _state(rng, empty=(2,))
    pred = target.copy()
    pred[..., layer_columns(NAMES, "temp")[2]] += 5.0  # error only in the empty bottom layer
    loss = hycom_wmse(_std(pred, stub), _std(target, stub), torch.ones(len(NAMES)))
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    pred[..., layer_columns(NAMES, "temp")[0]] += 5.0
    assert float(hycom_wmse(_std(pred, stub), _std(target, stub), torch.ones(len(NAMES)))) > 0


def test_penalties_vanish_on_the_truth_and_see_inversions():
    rng = np.random.default_rng(3)
    stub = _Stub({"thickness_weighted": True, "density": 1.0, "stability": 1.0})
    CONTEXT.configure(stub)
    target = _std(_state(rng), stub)
    ones = torch.ones(len(NAMES))
    assert float(hycom_wmse(target, target, ones)) == pytest.approx(0.0, abs=1e-6)
    inverted = _state(rng)
    inverted[..., layer_columns(NAMES, "temp")] = 5 + 5 * np.arange(L)  # warmer below: unstable
    pred = _std(inverted, stub)
    off = dict(stub.config["physics"], density=0.0, stability=0.0)
    CONTEXT.settings |= off
    plain = float(hycom_wmse(pred, target, ones))
    CONTEXT.settings |= {"stability": 1.0}
    assert float(hycom_wmse(pred, target, ones)) > plain


def test_group_weights():
    w = group_weights(NAMES)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["srfhgt"] == pytest.approx(1 / 9) and w["temp_k01"] == pytest.approx(1 / (9 * L))


def test_arm_configs_load_in_neural_lam(tmp_path):
    from neural_lam.config import NeuralLAMConfig
    from test_stack import _run, write_pack

    from hycom_emulator.datastore import HycomDatastore

    rng = np.random.default_rng(4)
    pack = write_pack(tmp_path / "p", *_run(rng, False, rng.normal(size=(10, 12, 1))))
    for arm in ("control", "reweight", "penalties"):
        d = tmp_path / arm
        d.mkdir()
        (d / "b00.yaml").write_text(f"zarr: {pack}\nsplits:\n  train: [2025-06-01, 2025-06-06]\n  val: [2025-06-05, 2025-06-09]\n  test: [2025-06-05, 2025-06-10]\n")
        (d / "nlam.yaml").write_text("datastore:\n  kind: hycom\n  config_path: b00.yaml\n")
        args = apply_arm(arm, d / "b00.yaml", d / "nlam.yaml")
        cfg = NeuralLAMConfig.from_yaml_file(d / "nlam.yaml")
        ds = HycomDatastore(d / "b00.yaml")
        if arm == "control":
            assert args == ["--model", "graph_lam", "--loss", "wmse"] and "physics" not in ds.config
        else:
            assert args == ["--model", "hycom_graph_lam", "--loss", "hycom_wmse"]
            assert sum(cfg.training.state_feature_weighting.weights.values()) == pytest.approx(1.0)
            assert ds.config["physics"]["density"] == (1 / 9 if arm == "penalties" else 0.0)
