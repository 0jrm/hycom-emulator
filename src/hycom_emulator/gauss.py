"""A GraphLAM with a Gaussian head, trained with the closed-form Gaussian CRPS (the Gaussian-head arm of the reanalysis emulator).

neural-lam's `--output_std` doubles the output map's last layer: per grid point the step predictor emits the mean
change (rows 0..F-1, rescaled by the one-day change std as usual) and sigma = softplus(rows F..2F-1). sigma is not
rescaled, and the loss compares it with target - mean, both standardized by the state mean and std, so sigma is in
standardized state units: sigma * state_std is the forecast std in physical units. It is the std of each step's
forecast against the truth at that lead, and the step predictor does not know the lead.

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

    train_model --model graph_lam --output_std --loss wcrps_gauss --init_from <graph_lam.ckpt> [rea_train flags]

Validation and test also log `{val,test}_wmse` (wmse of the mean, comparable to a deterministic run's val_mean_loss)
and `{val,test}_spread_skill` (the rms of sigma over the rmse of the mean, each weighted like wmse; 1 is calibrated).
Importing this module registers `wcrps_gauss`.
"""

from __future__ import annotations

import argparse
import math

import torch
from neural_lam import metrics
from neural_lam.loss_weighting import get_state_feature_weighting
from neural_lam.models import ForecasterModule

INIT_STD = 0.55
LOSS = "wcrps_gauss"


def gaussian_crps(mean: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """CRPS of N(mean, sigma^2) against y, entrywise, in the units of y."""
    z = (y - mean) / sigma
    pdf = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    return sigma * (z * (2 * torch.special.ndtr(z) - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def wcrps_gauss(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True, *, weight):
    """pred, target, pred_std (..., N, F) standardized; weight (F,) per_var_std. Reduced as neural-lam's metrics.
    weight is keyword-only and required, so a plain ForecasterModule, which would call this with sigma alone, fails."""
    return metrics.mask_and_reduce_metric(gaussian_crps(pred, pred_std, target) / weight, mask, average_grid, sum_vars)


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
    """Load a deterministic checkpoint into the Gaussian head: mean rows copied, sigma rows at INIT_STD change stds."""
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    predictor = module.forecaster.predictor
    head = f"forecaster.predictor.output_map.{len(predictor.output_map) - 1}."
    weight, bias = state[head + "weight"], state[head + "bias"]
    sigma0 = INIT_STD * predictor.diff_std.cpu()
    state[head + "weight"] = torch.cat([weight, torch.zeros_like(weight)])
    state[head + "bias"] = torch.cat([bias, sigma0 + torch.log(-torch.expm1(-sigma0))])  # softplus^-1
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
    """(--init_from, the rest) of a `--loss wcrps_gauss` command line. Exits on an inconsistent set."""
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--init_from", default=None)
    p.add_argument("--model", default="graph_lam")
    ours, rest = p.parse_known_args(argv)
    if ours.model != "graph_lam":
        raise SystemExit(f"--loss {LOSS} takes --model graph_lam, not {ours.model}")
    if "--output_std" not in rest:
        raise SystemExit(f"--loss {LOSS} needs --output_std")
    return ours.init_from, rest + ["--model", ours.model]


metrics.DEFINED_METRICS[LOSS] = wcrps_gauss
