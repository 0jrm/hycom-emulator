"""rea_train: the arm flags and ReaForecaster, with a one-parameter predictor (next = a * prev) on a stub datastore."""

import neural_lam.train_model as tm
import numpy as np
import pytest
import torch
import xarray as xr
from neural_lam.models import ARForecaster

from hycom_emulator import rea_train
from hycom_emulator.rea_train import Options, ReaForecaster, split_args

N, F = 40, 3
BOUNDARY = np.arange(N) >= 30
DIFF_STD = np.array([0.5, 1.0, 2.0])


class _Store:
    """The parts of a datastore ARForecaster and ReaForecaster read."""

    boundary_mask = xr.DataArray(BOUNDARY.astype(np.int8))

    def get_standardization_dataarray(self, category):
        return xr.Dataset({"state_diff_std_standardized": ("f", DIFF_STD)})


class _Linear(torch.nn.Module):
    predicts_std = False

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.9, dtype=torch.float64))
        self.seen = []

    def forward(self, prev_state, prev_prev_state, forcing):
        self.seen.append(prev_state.detach().clone())
        return self.a * prev_state, None


def _inputs(rng, b=2, t=4):
    init = torch.tensor(rng.normal(size=(b, 2, N, F)))
    forcing = torch.zeros(b, t, N, 0, dtype=torch.float64)
    boundary = torch.tensor(rng.normal(size=(b, t, N, F)))
    return init, forcing, boundary


def _forecaster(cls=ReaForecaster, **kw):
    return cls(_Linear(), _Store(), **kw).double()


def test_pushforward_steps_carry_no_gradient_and_the_next_step_sees_one_step():
    rng = np.random.default_rng(0)
    fc = _forecaster(pushforward=2).train()
    pred, std = fc(*_inputs(rng))
    assert std is None and pred.shape == (2, 4, N, F)
    (early,) = torch.autograd.grad(pred[:, :2].sum(), fc.predictor.a, retain_graph=True)
    assert early == 0, "the pushforward steps add no gradient"
    (grad,) = torch.autograd.grad(pred[:, 2].sum(), fc.predictor.a)
    interior = torch.tensor(~BOUNDARY)[:, None]
    assert torch.allclose(grad, (pred[:, 1] * interior).sum()), "d pred_K / d a is the state at K - 1, held fixed"
    full = _forecaster(cls=ReaForecaster).train()
    pred_full, _ = full(*_inputs(np.random.default_rng(0)))
    assert torch.allclose(pred_full, pred)
    (grad_full,) = torch.autograd.grad(pred_full[:, 2].sum(), full.predictor.a)
    assert not torch.allclose(grad_full, grad), "without pushforward the chain adds the earlier steps"


def test_eval_mode_is_ar_forecaster_exactly_and_ignores_the_switches():
    rng = np.random.default_rng(1)
    inputs = _inputs(rng)
    base = _forecaster(cls=ARForecaster).eval()
    arm = _forecaster(pushforward=3, input_noise=0.5, checkpoint_steps=True).eval()
    assert torch.equal(arm(*inputs)[0], base(*inputs)[0])
    assert torch.equal(arm.predictor.seen[0], inputs[0][:, 1]), "no noise in eval"


def test_checkpointed_steps_give_the_same_values_and_gradients():
    rng = np.random.default_rng(2)
    inputs = _inputs(rng)
    plain, ckpt = _forecaster(pushforward=1).train(), _forecaster(pushforward=1, checkpoint_steps=True).train()
    p0, p1 = plain(*inputs)[0], ckpt(*inputs)[0]
    assert torch.equal(p0, p1)
    g0 = torch.autograd.grad((p0**2).mean(), plain.predictor.a)
    g1 = torch.autograd.grad((p1**2).mean(), ckpt.predictor.a)
    assert torch.allclose(g0[0], g1[0])


def test_input_noise_touches_interior_points_only_at_the_change_std_scale():
    rng = np.random.default_rng(3)
    init, forcing, boundary = _inputs(rng, b=400, t=1)
    fc = _forecaster(input_noise=0.5).train()
    torch.manual_seed(0)
    fc(init, forcing, boundary)
    noise = fc.predictor.seen[0] - init[:, 1]
    assert torch.equal(noise[:, BOUNDARY], torch.zeros_like(noise[:, BOUNDARY]))
    std = noise[:, ~BOUNDARY].reshape(-1, F).std(0).numpy()
    assert np.allclose(std, 0.5 * DIFF_STD, rtol=0.05)


def test_pushforward_needs_a_longer_rollout():
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="pushforward"):
        _forecaster(pushforward=4).train()(*_inputs(rng, t=4))


def test_split_args_takes_our_flags_and_leaves_neural_lams():
    opts, rest = split_args(["--config_path", "n.yaml", "--loss", "rea_wmse", "--mean_penalty", "0.0016", "--mean_scales", "s.npz",
                             "--amse", "0.1", "--pushforward", "1", "--input_noise", "0.2", "--checkpoint_steps", "--lr", "1e-4"])
    assert opts == Options(0.0016, rea_train.Path("s.npz"), 0.1, 1, 0.2, True)
    assert rest == ["--config_path", "n.yaml", "--loss", "rea_wmse", "--lr", "1e-4"]
    assert split_args(["--log_domain_means", "--val_steps_to_log", "1", "4"]) == (Options(log_domain_means=True), ["--val_steps_to_log", "1", "4"])
    assert split_args(["--loss", "rea_wmse", "--mean_penalty", "0"])[0].mean_penalty == 0.0
    assert split_args(["--loss", "wmse"]) == (Options(), ["--loss", "wmse"])


@pytest.mark.parametrize("argv", [
    ["--loss", "rea_wmse"],
    ["--mean_penalty", "0"],
    ["--loss", "wmse", "--amse", "0.1"],
    ["--loss", "rea_wmse", "--mean_penalty", "0.1"],
    ["--pushforward", "-1"],
])
def test_split_args_refuses_an_inconsistent_set(argv):
    with pytest.raises(SystemExit):
        split_args(argv)


def test_install_swaps_the_forecaster_class(monkeypatch):
    monkeypatch.setattr(tm, "ARForecaster", tm.ARForecaster)
    monkeypatch.setattr(tm, "ForecasterModule", tm.ForecasterModule)
    rea_train.install(Options(), ["--config_path", "n.yaml"])
    assert tm.ARForecaster is ARForecaster and tm.ForecasterModule is rea_train.ForecasterModule
    rea_train.install(Options(log_domain_means=True), ["--config_path", "n.yaml"])
    assert tm.ForecasterModule is rea_train.ReaForecasterModule and tm.ARForecaster is ARForecaster
    rea_train.install(Options(pushforward=1), ["--config_path", "n.yaml"])
    fc = tm.ARForecaster(_Linear(), _Store())
    assert isinstance(fc, ReaForecaster) and fc.pushforward == 1
    assert not dict(fc.state_dict(keep_vars=True)).keys() - {"predictor.a"}, "no new weights in the checkpoint"
