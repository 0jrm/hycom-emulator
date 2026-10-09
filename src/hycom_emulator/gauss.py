"""A GraphLAM with a Gaussian head, trained with the closed-form Gaussian CRPS (the Gaussian-head arm of the reanalysis emulator).

neural-lam's `--output_std` doubles the output map's last layer: per grid point the step predictor emits the mean
change (rows 0..F-1, rescaled by the one-day change std as usual) and sigma = softplus(rows F..2F-1). sigma is not
rescaled, and the loss compares it with target - mean, both standardized by the state mean and std, so sigma is in
standardized state units: sigma * state_std is the forecast std in physical units. It is the std of each step's
forecast against the truth at that lead, and without --std_feedback the step predictor does not know the lead.

    wcrps_gauss = sum_f mean_interior CRPS(N(mu, sigma^2), y) / per_var_std_f
    CRPS(N(mu, sigma^2), y) = sigma (z (2 Phi(z) - 1) + 2 phi(z) - 1 / sqrt(pi)),  z = (y - mu) / sigma

per (batch, step), then neural-lam's mean over batch and steps. per_var_std is the per-channel weight of wmse and of
the crps arm's afcrps: state_diff_std_standardized / sqrt(feature weight), with uniform feature weights 1 / F, so
sqrt(F) times the one-day change std in standardized units. As sigma -> 0 the CRPS tends to |y - mu|, so
wcrps_gauss tends to wmae.

neural-lam hands the loss the predicted sigma in the slot where it otherwise passes per_var_std, so the weight
travels another way: GaussForecasterModule keeps per_var_std as a buffer and binds `self.loss` to wcrps_gauss with
that buffer. neural-lam calls `self.loss` in training, validation, test and for the test's spatial loss maps; the
buffer follows the module to its device, and nothing has to be configured by each entry point (as rea_wmse is).

`--init_from <graph_lam.ckpt>` loads every weight of a deterministic checkpoint. The output map's last layer gets the
checkpoint's rows for the mean and zero weights with bias softplus^-1(INIT_STD x state_diff_std_standardized) for
sigma, so the mean forecast starts as the checkpoint's exactly and sigma starts at the one-day error of emu-rea-s2b
(its 1-day val wmse 0.30 is the channel mean of (rmse / change std)^2, sqrt 0.55). Training starts at epoch 0. With
`--load` as well (later stages of train_rea.sh, the test pass) Lightning's restore replaces these weights afterwards.

    train_model --model graph_lam --output_std --loss wcrps_gauss [--std_feedback] --init_from <graph_lam.ckpt> [rea_train flags]

`--std_feedback` makes the uncertainty part of the state: the step predictor (StdFeedbackGraphLAM, model name
std_feedback_graph_lam in the checkpoint) also reads the sigma it predicted for the previous step, so
sigma_k = g(state_k, sigma_{k-1}, forcing) can learn how the error grows and moves. The input is
log1p(sigma / change std) per channel: 0 for a true state (step 1, the boundary band), about 0.4 at the one-day
error, and logarithmic for the multi-day sigmas, which would otherwise dwarf the standardized state inputs.
hycom_emulator.rea_train.ReaForecaster carries sigma between steps; the routing installs it for this arm. The 91
new input columns of the grid embedder start at zero, so from --init_from the mean is still the checkpoint's.

Validation and test also log `{val,test}_wmse` (wmse of the mean, comparable to a deterministic run's val_mean_loss)
and `{val,test}_spread_skill` (the rms of sigma over the rmse of the mean, each weighted like wmse; 1 is calibrated).
Importing this module registers `wcrps_gauss` and `std_feedback_graph_lam`.
"""

from __future__ import annotations

import argparse
import math

import torch
from neural_lam import metrics, utils
from neural_lam.loss_weighting import get_state_feature_weighting
from neural_lam.models import MODELS, ForecasterModule
from neural_lam.models.step_predictors.graph.graph_lam import GraphLAM

INIT_STD = 0.55
LOSS = "wcrps_gauss"
FEEDBACK_MODEL = "std_feedback_graph_lam"


def gaussian_crps(mean: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """CRPS of N(mean, sigma^2) against y, entrywise, in the units of y."""
    z = (y - mean) / sigma
    pdf = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    return sigma * (z * (2 * torch.special.ndtr(z) - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def wcrps_gauss(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True, *, weight):
    """pred, target, pred_std (..., N, F) standardized; weight (F,) per_var_std. Reduced as neural-lam's metrics.
    weight is keyword-only and required, so a plain ForecasterModule, which would call this with sigma alone, fails."""
    return metrics.mask_and_reduce_metric(gaussian_crps(pred, pred_std, target) / weight, mask, average_grid, sum_vars)


class StdFeedbackGraphLAM(GraphLAM):
    """GraphLAM with a Gaussian head whose grid input also carries log1p(previous sigma / change std), placed after the
    forcing and before the static features. Called by ReaForecaster with that sigma as a fourth argument."""

    std_feedback = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.output_std:
            raise ValueError("std feedback needs --output_std")
        self.grid_input_dim += self.grid_output_dim // 2
        self.grid_embedder = utils.make_mlp([self.grid_input_dim] + self.mlp_blueprint_end)

    def forward(self, prev_state, prev_prev_state, forcing, prev_std):
        return super().forward(prev_state, prev_prev_state, torch.cat([forcing, torch.log1p(prev_std / self.diff_std)], dim=-1))


class GaussForecasterModule(ForecasterModule):
    def __init__(self, forecaster, config, datastore, **kwargs):
        super().__init__(forecaster=forecaster, config=config, datastore=datastore, **kwargs)
        if not self.forecaster.predicts_std:
            raise ValueError(f"--loss {LOSS} needs a predictor with --output_std")
        if self.loss is not wcrps_gauss:
            raise ValueError(f"a Gaussian head trains with --loss {LOSS}")
        stats = datastore.get_standardization_dataarray("state")
        diff_std = torch.tensor(stats.state_diff_std_standardized.values, dtype=torch.float32)
        weights = torch.tensor(get_state_feature_weighting(config=config, datastore=datastore), dtype=torch.float32)
        del self.per_var_std
        self.register_buffer("per_var_std", diff_std / weights.sqrt(), persistent=False)
        self.loss = self.weighted_crps

    def weighted_crps(self, pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
        return wcrps_gauss(pred, target, pred_std, mask, average_grid, sum_vars, weight=self.per_var_std)

    def _compute_prediction_and_loss(self, batch):
        out = super()._compute_prediction_and_loss(batch)
        if self._trainer is not None and not self.training:
            prediction, target, sigma, _ = out
            weigh = lambda x: metrics.mask_and_reduce_metric(x / self.per_var_std**2, self.interior_mask_bool, True, True).mean()  # noqa: E731
            mse = weigh((prediction - target) ** 2)
            phase = "test" if self._trainer.testing else "val"
            self.log_dict({f"{phase}_wmse": mse, f"{phase}_spread_skill": (weigh(sigma**2) / mse).sqrt()},
                          on_epoch=True, sync_dist=True, batch_size=target.shape[0])
        return out


def init_from_deterministic(module: GaussForecasterModule, path: str) -> None:
    """Load a deterministic checkpoint into the Gaussian head: mean rows copied, sigma rows at INIT_STD change stds,
    and for std feedback zero grid-embedder columns for the sigma input."""
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    predictor = module.forecaster.predictor
    head = f"forecaster.predictor.output_map.{len(predictor.output_map) - 1}."
    weight, bias = state[head + "weight"], state[head + "bias"]
    sigma0 = INIT_STD * predictor.diff_std.cpu()
    state[head + "weight"] = torch.cat([weight, torch.zeros_like(weight)])
    state[head + "bias"] = torch.cat([bias, sigma0 + torch.log(-torch.expm1(-sigma0))])  # softplus^-1
    if getattr(predictor, "std_feedback", False):
        key = "forecaster.predictor.grid_embedder.0.weight"
        w = state[key]
        at = w.shape[1] - predictor.grid_static_features.shape[1]
        state[key] = torch.cat([w[:, :at], w.new_zeros(w.shape[0], weight.shape[0]), w[:, at:]], dim=1)
    module.load_state_dict(state, strict=True)


def module_factory(init_from: str | None):
    """What train_model calls in place of ForecasterModule (it passes keywords only)."""

    def build(**kwargs):
        module = GaussForecasterModule(**kwargs)
        if init_from:
            init_from_deterministic(module, init_from)
        return module

    return build


def split_args(argv: list[str]) -> tuple[str | None, list[str]]:
    """(--init_from, the rest) of a `--loss wcrps_gauss` command line; --std_feedback becomes the model name, which
    the checkpoint keeps for --load and the evaluators. Exits on an inconsistent set."""
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--init_from", default=None)
    p.add_argument("--model", default="graph_lam")
    p.add_argument("--std_feedback", action="store_true")
    ours, rest = p.parse_known_args(argv)
    if ours.model != "graph_lam":
        raise SystemExit(f"--loss {LOSS} takes --model graph_lam, not {ours.model}")
    if "--output_std" not in rest:
        raise SystemExit(f"--loss {LOSS} needs --output_std")
    return ours.init_from, rest + ["--model", FEEDBACK_MODEL if ours.std_feedback else ours.model]


MODELS[FEEDBACK_MODEL] = StdFeedbackGraphLAM
metrics.DEFINED_METRICS[LOSS] = wcrps_gauss
